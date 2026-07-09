// scirust-core/src/matrix/view.rs
//
// Vues de matrices sans allocation.
// MatrixView / MatrixViewMut permettent d'accéder à des sous-matrices
// (blocs, lignes, colonnes) en partageant la mémoire du parent.
//
// Modèle mémoire :
//   data[row * row_stride + col * col_stride]
//   Layout row-major  : row_stride = cols,  col_stride = 1
//   Layout col-major  : row_stride = 1,     col_stride = rows
//   Sous-matrice      : strides identiques au parent, ptr décalé

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

// ------------------------------------------------------------------ //
//  Trait de base partagé                                              //
// ------------------------------------------------------------------ //

pub trait MatrixShape {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    #[inline]
    fn len(&self) -> usize {
        self.rows() * self.cols()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
}

// ------------------------------------------------------------------ //
//  MatrixView<'a, T> — vue immuable                                   //
// ------------------------------------------------------------------ //

#[derive(Clone, Copy)]
pub struct MatrixView<'a, T> {
    ptr: *const T,
    rows: usize,
    cols: usize,
    row_stride: usize, // sauts en mémoire entre deux lignes
    col_stride: usize, // sauts en mémoire entre deux colonnes
    _marker: PhantomData<&'a T>,
}

// SAFETY : T: Sync implique MatrixView<T>: Send + Sync
unsafe impl<'a, T: Sync> Send for MatrixView<'a, T> {}
unsafe impl<'a, T: Sync> Sync for MatrixView<'a, T> {}

impl<'a, T> MatrixView<'a, T> {
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    #[inline]
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    /// Crée une vue row-major standard sur un slice.
    /// Précondition : data.len() >= rows * cols
    #[inline]
    pub fn from_slice(data: &'a [T], rows: usize, cols: usize) -> Self {
        // `rows * cols` doit être vérifié : en release le produit wrappe, donc
        // des dimensions pathologiques pourraient wrapper vers un petit produit,
        // passer l'assert sur un slice trop court, puis produire une vue dont les
        // accès lisent hors bornes.
        let need = rows
            .checked_mul(cols)
            .expect("MatrixView::from_slice: rows * cols overflows usize");
        assert!(data.len() >= need, "slice trop petit pour la vue");
        Self {
            ptr: data.as_ptr(),
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
            _marker: PhantomData,
        }
    }

    /// Crée une vue avec strides personnalisés (utile pour col-major, tiles…)
    ///
    /// # Safety
    /// L'appelant garantit que tous les indices (r*row_stride + c*col_stride)
    /// restent dans les limites de la mémoire pointée.
    #[inline]
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn from_raw_parts(
        ptr: *const T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        Self {
            ptr,
            rows,
            cols,
            row_stride,
            col_stride,
            _marker: PhantomData,
        }
    }

    /// Sous-vue [row_start..row_start+nrows, col_start..col_start+ncols]
    /// Zéro allocation — renvoie un pointeur décalé avec les mêmes strides.
    #[inline]
    pub fn subview(
        &self,
        row_start: usize,
        nrows: usize,
        col_start: usize,
        ncols: usize,
    ) -> MatrixView<'a, T> {
        assert!(
            row_start + nrows <= self.rows,
            "sous-vue hors bornes (lignes)"
        );
        assert!(
            col_start + ncols <= self.cols,
            "sous-vue hors bornes (colonnes)"
        );
        unsafe {
            MatrixView::from_raw_parts(
                self.ptr
                    .add(row_start * self.row_stride + col_start * self.col_stride),
                nrows,
                ncols,
                self.row_stride,
                self.col_stride,
            )
        }
    }

    /// Vue sur la ligne `r` (slice de longueur cols si col-majeur = 1)
    #[inline]
    pub fn row(&self, r: usize) -> MatrixView<'a, T> {
        self.subview(r, 1, 0, self.cols)
    }

    /// Vue sur la colonne `c`
    #[inline]
    pub fn col(&self, c: usize) -> MatrixView<'a, T> {
        self.subview(0, self.rows, c, 1)
    }

    /// Accès en lecture avec vérification de bornes (API sûre : panique en
    /// release comme en debug si l'index est hors bornes, au lieu de produire un
    /// accès mémoire hors limites — UB — via le pointeur brut).
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> &T {
        assert!(
            r < self.rows && c < self.cols,
            "index ({r}, {c}) hors bornes pour une vue {}x{}",
            self.rows,
            self.cols
        );
        // SAFETY: bornes vérifiées juste au-dessus.
        unsafe { self.get_unchecked(r, c) }
    }

    /// Accès brut en lecture, sans vérification de bornes — pour les boucles
    /// internes déjà prouvées dans les limites (GEMM/GEMV).
    ///
    /// # Safety
    /// L'appelant garantit `r < rows` et `c < cols`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        unsafe { &*self.ptr.add(r * self.row_stride + c * self.col_stride) }
    }

    /// Itérateur sur les lignes (chaque ligne est une `MatrixView<T>`)
    pub fn row_iter(&self) -> impl Iterator<Item = MatrixView<'a, T>> + '_ {
        (0..self.rows).map(move |r| self.row(r))
    }

    /// Slice contiguë sur une ligne (uniquement si col_stride == 1)
    pub fn row_slice(&self, r: usize) -> Option<&'a [T]> {
        if self.col_stride == 1
        {
            Some(unsafe {
                std::slice::from_raw_parts(self.ptr.add(r * self.row_stride), self.cols)
            })
        }
        else
        {
            None
        }
    }
}

impl<'a, T> MatrixShape for MatrixView<'a, T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a, T> Index<(usize, usize)> for MatrixView<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, (r, c): (usize, usize)) -> &T {
        self.get(r, c)
    }
}

// ------------------------------------------------------------------ //
//  MatrixViewMut<'a, T> — vue mutable                                 //
// ------------------------------------------------------------------ //

pub struct MatrixViewMut<'a, T> {
    ptr: *mut T,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
    _marker: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: Send> Send for MatrixViewMut<'a, T> {}

impl<'a, T> MatrixViewMut<'a, T> {
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    #[inline]
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }

    #[inline]
    /// # Safety
    /// Pointer must be valid.
    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        rows: usize,
        cols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        Self {
            ptr,
            rows,
            cols,
            row_stride,
            col_stride,
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn from_slice(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let need = rows
            .checked_mul(cols)
            .expect("MatrixViewMut::from_slice: rows * cols overflows usize");
        assert!(data.len() >= need, "slice trop petit pour la vue");
        Self {
            ptr: data.as_mut_ptr(),
            rows,
            cols,
            row_stride: cols,
            col_stride: 1,
            _marker: PhantomData,
        }
    }

    /// Vue immuable à partir d'une vue mutable
    #[inline]
    pub fn as_view(&self) -> MatrixView<'_, T> {
        unsafe {
            MatrixView::from_raw_parts(
                self.ptr as *const T,
                self.rows,
                self.cols,
                self.row_stride,
                self.col_stride,
            )
        }
    }

    /// Sous-vue mutable — même logique que MatrixView::subview
    #[inline]
    pub fn subview_mut(
        &mut self,
        row_start: usize,
        nrows: usize,
        col_start: usize,
        ncols: usize,
    ) -> MatrixViewMut<'_, T> {
        assert!(row_start + nrows <= self.rows);
        assert!(col_start + ncols <= self.cols);
        unsafe {
            MatrixViewMut {
                ptr: self
                    .ptr
                    .add(row_start * self.row_stride + col_start * self.col_stride),
                rows: nrows,
                cols: ncols,
                row_stride: self.row_stride,
                col_stride: self.col_stride,
                _marker: PhantomData,
            }
        }
    }

    /// Accès en lecture (immuable) avec vérification de bornes.
    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> &T {
        assert!(
            r < self.rows && c < self.cols,
            "index ({r}, {c}) hors bornes pour une vue {}x{}",
            self.rows,
            self.cols
        );
        // SAFETY: bornes vérifiées juste au-dessus.
        unsafe { &*self.ptr.add(r * self.row_stride + c * self.col_stride) }
    }

    /// Accès brut en lecture sans vérification de bornes.
    ///
    /// # Safety
    /// L'appelant garantit `r < rows` et `c < cols`.
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, r: usize, c: usize) -> &T {
        unsafe { &*self.ptr.add(r * self.row_stride + c * self.col_stride) }
    }

    /// Accès mutable avec vérification de bornes (panique en release aussi).
    #[inline(always)]
    pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
        assert!(
            r < self.rows && c < self.cols,
            "index ({r}, {c}) hors bornes pour une vue {}x{}",
            self.rows,
            self.cols
        );
        // SAFETY: bornes vérifiées juste au-dessus.
        unsafe { self.get_unchecked_mut(r, c) }
    }

    /// Accès mutable sans vérification de bornes — boucles internes prouvées.
    ///
    /// # Safety
    /// L'appelant garantit `r < rows` et `c < cols`.
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, r: usize, c: usize) -> &mut T {
        unsafe { &mut *self.ptr.add(r * self.row_stride + c * self.col_stride) }
    }

    /// Slice contiguë mutable sur une ligne (uniquement si col_stride == 1)
    pub fn row_slice_mut(&mut self, r: usize) -> Option<&mut [T]> {
        if self.col_stride == 1
        {
            Some(unsafe {
                std::slice::from_raw_parts_mut(self.ptr.add(r * self.row_stride), self.cols)
            })
        }
        else
        {
            None
        }
    }
}

impl<'a, T> MatrixShape for MatrixViewMut<'a, T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
}

impl<'a, T> Index<(usize, usize)> for MatrixViewMut<'a, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, (r, c): (usize, usize)) -> &T {
        self.get(r, c)
    }
}

impl<'a, T> IndexMut<(usize, usize)> for MatrixViewMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut T {
        self.get_mut(r, c)
    }
}

// ------------------------------------------------------------------ //
//  Tests                                                              //
// ------------------------------------------------------------------ //
#[cfg(test)]
mod tests {
    use super::*;

    fn make_4x4() -> Vec<f64> {
        (0..16).map(|x| x as f64).collect()
    }

    #[test]
    fn test_view_access() {
        let data = make_4x4();
        let view = MatrixView::from_slice(&data, 4, 4);
        assert_eq!(view[(0, 0)], 0.0);
        assert_eq!(view[(1, 2)], 6.0); // 1*4 + 2 = 6
        assert_eq!(view[(3, 3)], 15.0);
    }

    // Regression: a SAFE constructor + SAFE index must never read out of bounds
    // in release. Previously `get`/`Index` only `debug_assert!`ed, so this read
    // was UB in a release build. It must now panic (bounds-checked) in every
    // profile.
    #[test]
    #[should_panic(expected = "hors bornes")]
    fn get_out_of_range_panics_not_ub() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let view = MatrixView::from_slice(&data, 2, 2);
        let _ = view[(9, 9)]; // 9 >= rows/cols -> must panic, not OOB-read
    }

    #[test]
    #[should_panic(expected = "hors bornes")]
    fn get_mut_out_of_range_panics_not_ub() {
        let mut data = vec![1.0f64, 2.0, 3.0, 4.0];
        let mut view = MatrixViewMut::from_slice(&mut data, 2, 2);
        *view.get_mut(5, 0) = 0.0; // must panic, not OOB-write
    }

    // Regression: `rows * cols` must be checked so a wrapping product cannot pass
    // the slice-length assert and yield a view over a too-short slice.
    #[test]
    #[should_panic(expected = "overflows usize")]
    fn from_slice_rejects_dimension_overflow() {
        let data = vec![0.0f64; 4];
        // rows * cols wraps to a small value in release without the checked_mul.
        let _ = MatrixView::from_slice(&data, usize::MAX, 2);
    }

    #[test]
    fn test_subview_no_alloc() {
        let data = make_4x4();
        let view = MatrixView::from_slice(&data, 4, 4);
        // Sous-matrice 2x2 à partir de (1,1)
        let sub = view.subview(1, 2, 1, 2);
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(sub[(0, 0)], 5.0); // data[1*4+1]
        assert_eq!(sub[(1, 1)], 10.0); // data[2*4+2]
    }

    #[test]
    fn test_row_slice() {
        let data = make_4x4();
        let view = MatrixView::from_slice(&data, 4, 4);
        let row2 = view.row_slice(2).unwrap();
        assert_eq!(row2, &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_view_mut() {
        let mut data = make_4x4();
        let mut view = MatrixViewMut::from_slice(&mut data, 4, 4);
        view[(2, 2)] = 999.0;
        assert_eq!(view[(2, 2)], 999.0);
        assert_eq!(data[10], 999.0); // 2*4+2 = 10
    }

    #[test]
    fn test_col_view() {
        let data = make_4x4();
        let view = MatrixView::from_slice(&data, 4, 4);
        let col1 = view.col(1);
        assert_eq!(col1.shape(), (4, 1));
        assert_eq!(col1[(0, 0)], 1.0);
        assert_eq!(col1[(1, 0)], 5.0);
        assert_eq!(col1[(2, 0)], 9.0);
    }
}
