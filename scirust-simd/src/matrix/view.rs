// Stub matrix view types for scirust-simd.

#[derive(Debug, Clone)]
pub struct MatrixView<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
    col_stride: usize,
}

impl<'a, T: Copy> MatrixView<'a, T> {
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self {
            data,
            rows,
            cols,
            col_stride: 1,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row_slice(&self, i: usize) -> Option<&[T]> {
        if i >= self.rows
        {
            return None;
        }
        let start = i * self.cols * self.col_stride;
        let end = start + self.cols;
        Some(&self.data[start..end])
    }
}

#[derive(Debug)]
pub struct MatrixViewMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T: Copy> MatrixViewMut<'a, T> {
    pub fn new(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major read access to row `i` (contiguous `cols` elements).
    pub fn row_slice(&self, i: usize) -> Option<&[T]> {
        if i >= self.rows
        {
            return None;
        }
        let start = i * self.cols;
        Some(&self.data[start..start + self.cols])
    }

    /// Row-major mutable access to row `i` (contiguous `cols` elements).
    pub fn row_slice_mut(&mut self, i: usize) -> Option<&mut [T]> {
        if i >= self.rows
        {
            return None;
        }
        let start = i * self.cols;
        Some(&mut self.data[start..start + self.cols])
    }
}
