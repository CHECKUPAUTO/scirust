# Faire évoluer un algorithme depuis votre LLM (serveur MCP)

Ce guide permet à un **utilisateur non-technique** de demander à son LLM, en
langage naturel, de **se connecter à scirust** et de faire évoluer un
algorithme à partir d'exemples qu'il fournit.

Le mécanisme est le **Model Context Protocol (MCP)** : scirust expose un petit
serveur (`scirust-rsi-mcp`) que votre client LLM appelle comme un outil. Le
serveur fait évoluer un programme arithmétique (sur une entrée `x`) pour
reproduire vos exemples *entrée → sortie*, sous les garanties de scirust
(borné, élitiste/non-régressif, reproductible, sandboxé).

> ⚠️ Une configuration **unique** est nécessaire : un LLM ne peut se connecter à
> un outil externe qu'après l'avoir déclaré une fois dans un client compatible
> MCP (Claude Desktop, Claude Code, etc.). Un LLM purement web sans support MCP
> ne peut pas s'y connecter — voir la note en bas.

## 1. Compiler le serveur (une fois)

```sh
cargo build -p scirust-rsi --bin scirust-rsi-mcp --features mcp --release
# binaire produit : target/release/scirust-rsi-mcp
```

## 2. Déclarer le serveur dans votre client (une fois)

**Claude Code** (CLI) :

```sh
claude mcp add scirust -- /chemin/absolu/vers/target/release/scirust-rsi-mcp
```

**Claude Desktop** — ajoutez ceci à `claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "scirust": {
      "command": "/chemin/absolu/vers/target/release/scirust-rsi-mcp"
    }
  }
}
```

Redémarrez le client. L'outil `evolve_algorithm` est maintenant disponible.

## 3. La commande à coller dans votre LLM

Copiez-collez (en adaptant vos exemples) :

> Connecte-toi au serveur MCP « scirust » et utilise l'outil
> **`evolve_algorithm`** pour faire évoluer un programme qui reproduit ces
> exemples entrée → sortie :
>
> `1 → 2, 2 → 4, 3 → 6, 4 → 8`
>
> (autrement dit : doubler l'entrée). Donne-moi le programme évolué, son erreur,
> et vérifie qu'il colle à chaque exemple.

Le LLM appellera l'outil avec `examples = [[1,2],[2,4],[3,6],[4,8]]` et vous
répondra avec le programme trouvé (ici `x x +`, soit `2·x`), son erreur (0), et
le tableau de vérification.

### Faire évoluer *votre* algorithme de départ

Si vous avez déjà un algorithme et voulez l'améliorer, donnez-le comme point de
départ — le résultat ne sera **jamais pire** que le vôtre (sélection élitiste) :

> … utilise `evolve_algorithm` avec mes exemples `-1 → 2, 0 → 1, 1 → 2, 2 → 5`
> et **`seed_program` = "x x *"** comme point de départ. Améliore-le jusqu'à
> coller aux exemples.

## L'outil `evolve_algorithm`

| Argument | Requis | Défaut | Rôle |
|---|---|---|---|
| `examples` | oui | — | paires `[entrée, sortie]`, ex. `[[1,2],[2,4]]` |
| `seed_program` | non | `"x"` | programme de départ (notation polonaise inversée) |
| `max_iters` | non | `1500` | plafond d'itérations (toujours borné) |
| `samples` | non | `32` | candidats proposés par tour (best-of-n) |
| `seed` | non | `0` | graine RNG → run reproductible |

Le programme est exprimé en **notation polonaise inversée** sur `x`, avec les
jetons `x`, des nombres, et `+ - * /`. Exemple : `x x * 1 +` signifie `x² + 1`.

## Ce que scirust garantit (et ce qu'il ne fait pas)

- **Borné** : `max_iters` ⇒ l'évolution se termine toujours.
- **Non-régressif** : adoption élitiste ⇒ le résultat n'est jamais pire que le
  programme de départ.
- **Reproductible** : même `seed` ⇒ même résultat.
- **Sandboxé** : seul un interpréteur arithmétique fixe est exécuté — aucun code
  généré n'est lancé, aucun accès à la machine, pas d'auto-réécriture.

L'évolution tourne **localement et hors-ligne** dans le serveur : aucune clé API
n'est requise. Le LLM ne sert qu'à traduire votre demande en appel d'outil et à
vous expliquer le résultat.

## Limite & alternative

Un LLM web sans support MCP (p. ex. une interface de chat basique) ne peut pas se
connecter à un binaire local. Deux options dans ce cas :

1. Utiliser un client compatible MCP (Claude Desktop / Claude Code) — recommandé.
2. Héberger ce moteur derrière une **API HTTP** que le LLM peut appeler (non
   inclus ici ; le cœur `scirust_rsi::progevo::evolve` est directement
   réutilisable pour ça).
