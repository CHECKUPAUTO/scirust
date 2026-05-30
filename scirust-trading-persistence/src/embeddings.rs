//! Store SQLite des embeddings TRIBE Brain.
//!
//! Stockage : BLOB column avec les f32 little-endian. K-nearest est calculé
//! en Rust (cosine similarity) sur toute la table — OK jusqu'à ~10k events
//! (10k × 768 × 4 = 30 MB en mémoire). Au-delà, passer à sqlite-vec ou un
//! index ANN externe.
//!
//! Implemente `scirust_trading_news::EmbeddingStore` pour s'intégrer
//! transparentement avec le `ContextualEnricher`.

use crate::PersistenceResult;
use async_trait::async_trait;
use rusqlite::{params, Connection};
use scirust_trading_news::{cosine_similarity, EmbeddingStore, NewsError, NewsResult};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

/// DDL ajouté en complément des autres tables. Idempotent.
pub(crate) const DDL_EMBEDDINGS: &str = r#"
CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id    TEXT PRIMARY KEY,
    dim         INTEGER NOT NULL,
    model       TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    created_at  INTEGER NOT NULL
);
"#;

pub struct SqliteEmbeddingStore {
    conn: Arc<Mutex<Connection>>,
    model_tag: String,
}

impl SqliteEmbeddingStore {
    pub fn open(db_path: PathBuf, model_tag: impl Into<String>) -> PersistenceResult<Self> {
        let mut conn = Connection::open(&db_path)?;
        crate::schema::init(&mut conn)?;
        conn.execute_batch(DDL_EMBEDDINGS)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            model_tag: model_tag.into(),
        })
    }

    pub fn open_in_memory(model_tag: impl Into<String>) -> PersistenceResult<Self> {
        let mut conn = Connection::open_in_memory()?;
        crate::schema::init(&mut conn)?;
        conn.execute_batch(DDL_EMBEDDINGS)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            model_tag: model_tag.into(),
        })
    }

    /// Compte d'embeddings stockés.
    pub async fn count(&self) -> PersistenceResult<i64> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || -> PersistenceResult<i64> {
            let c = conn.blocking_lock();
            let n: i64 = c.query_row(
                "SELECT COUNT(*) FROM event_embeddings",
                [],
                |row| row.get(0),
            )?;
            Ok(n)
        })
        .await?
    }
}

#[async_trait]
impl EmbeddingStore for SqliteEmbeddingStore {
    async fn put(&self, event_id: Uuid, embedding: Vec<f32>) -> NewsResult<()> {
        let conn = Arc::clone(&self.conn);
        let model = self.model_tag.clone();
        let id = event_id.to_string();
        tokio::task::spawn_blocking(move || -> NewsResult<()> {
            let c = conn.blocking_lock();
            let dim = embedding.len() as i64;
            // Sérialise les f32 en LE bytes
            let mut buf = Vec::with_capacity(embedding.len() * 4);
            for f in &embedding {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            c.execute(
                "INSERT OR REPLACE INTO event_embeddings
                 (event_id, dim, model, embedding, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    id,
                    dim,
                    model,
                    buf,
                    chrono::Utc::now().timestamp_millis(),
                ],
            )
            .map_err(|e| NewsError::Http(format!("sqlite: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| NewsError::Http(format!("join: {e}")))??;
        Ok(())
    }

    async fn find_nearest(
        &self,
        query: &[f32],
        k: usize,
        exclude: Option<Uuid>,
    ) -> NewsResult<Vec<(Uuid, f64)>> {
        let conn = Arc::clone(&self.conn);
        let query = query.to_vec();
        let exclude_str = exclude.map(|u| u.to_string());
        let k = k.max(1);
        tokio::task::spawn_blocking(move || -> NewsResult<Vec<(Uuid, f64)>> {
            let c = conn.blocking_lock();
            let mut stmt = c
                .prepare("SELECT event_id, dim, embedding FROM event_embeddings")
                .map_err(|e| NewsError::Http(format!("sqlite: {e}")))?;
            let rows = stmt
                .query_map([], |row| {
                    let id: String = row.get(0)?;
                    let dim: i64 = row.get(1)?;
                    let blob: Vec<u8> = row.get(2)?;
                    Ok((id, dim, blob))
                })
                .map_err(|e| NewsError::Http(format!("sqlite: {e}")))?;

            let mut scored: Vec<(Uuid, f64)> = Vec::new();
            for r in rows {
                let (id_str, dim, blob) = r.map_err(|e| NewsError::Http(format!("sqlite: {e}")))?;
                if let Some(ex) = &exclude_str {
                    if &id_str == ex {
                        continue;
                    }
                }
                let id = Uuid::parse_str(&id_str).map_err(|e| NewsError::Parse(e.to_string()))?;
                let dim = dim as usize;
                if blob.len() != dim * 4 {
                    continue; // skip corrupted
                }
                let mut emb = Vec::with_capacity(dim);
                for chunk in blob.chunks_exact(4) {
                    let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    emb.push(f32::from_le_bytes(arr));
                }
                if emb.len() != query.len() {
                    continue; // dim mismatch
                }
                let sim = cosine_similarity(&query, &emb);
                scored.push((id, sim));
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            Ok(scored)
        })
        .await
        .map_err(|e| NewsError::Http(format!("join: {e}")))?
    }
}

// ─── HistoricalProvider impl on QueryApi ──────────────────────────────────

use scirust_trading_core::MarketReaction;
use scirust_trading_news::HistoricalProvider;

#[async_trait]
impl HistoricalProvider for crate::queries::QueryApi {
    async fn lookup_response(
        &self,
        tags: &[String],
        asset_symbol_canonical: &str,
        exchange: &str,
    ) -> NewsResult<Option<MarketReaction>> {
        self.historical_response_for_tags(tags, asset_symbol_canonical, exchange)
            .await
            .map_err(|e| NewsError::Http(format!("persistence: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trip_and_nearest() {
        let store = SqliteEmbeddingStore::open_in_memory("snowflake-arctic-v2").unwrap();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        let c = Uuid::new_v4();
        store.put(a, vec![1.0_f32, 0.0, 0.0]).await.unwrap();
        store.put(b, vec![0.95, 0.05, 0.0]).await.unwrap();
        store.put(c, vec![0.0, 0.0, 1.0]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 3);

        // query exactement aligné sur b → b doit être premier
        let query = vec![0.95_f32, 0.05, 0.0];
        let neighbors = store.find_nearest(&query, 3, None).await.unwrap();
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].0, b);
        assert!(neighbors[0].1 > 0.999);
        // c reste loin (orthogonal sur la 3e dim)
        assert_eq!(neighbors[2].0, c);
        assert!(neighbors[2].1 < 0.1);
    }

    #[tokio::test]
    async fn upsert_overwrites() {
        let store = SqliteEmbeddingStore::open_in_memory("test").unwrap();
        let id = Uuid::new_v4();
        store.put(id, vec![1.0, 0.0]).await.unwrap();
        store.put(id, vec![0.0, 1.0]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
        let n = store.find_nearest(&[0.0, 1.0], 1, None).await.unwrap();
        assert!(n[0].1 > 0.99);
    }

    #[tokio::test]
    async fn exclude_filters_self() {
        let store = SqliteEmbeddingStore::open_in_memory("t").unwrap();
        let a = Uuid::new_v4();
        let b = Uuid::new_v4();
        store.put(a, vec![1.0, 0.0]).await.unwrap();
        store.put(b, vec![0.99, 0.01]).await.unwrap();
        let n = store.find_nearest(&[1.0, 0.0], 5, Some(a)).await.unwrap();
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].0, b);
    }

    #[tokio::test]
    async fn empty_store_returns_empty() {
        let store = SqliteEmbeddingStore::open_in_memory("t").unwrap();
        let n = store.find_nearest(&[1.0, 0.0], 5, None).await.unwrap();
        assert!(n.is_empty());
    }
}
