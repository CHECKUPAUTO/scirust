//! Lie groups, Lie algebras and geometric algebra.

use crate::core::{Field, LieAlgebra, Ring};

/// Three-vector used by `so(3)` and `SE(3)`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

impl Vec3 {
    /// Construct a vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    /// Dot product.
    #[inline]
    pub fn dot(self, rhs: Self) -> f64 { self.x * rhs.x + self.y * rhs.y + self.z * rhs.z }
    /// Cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(self.y * rhs.z - self.z * rhs.y, self.z * rhs.x - self.x * rhs.z, self.x * rhs.y - self.y * rhs.x)
    }
    /// Euclidean norm.
    #[inline]
    pub fn norm(self) -> f64 { self.dot(self).sqrt() }
    /// Scalar multiplication.
    #[inline]
    pub fn scale(self, a: f64) -> Self { Self::new(a * self.x, a * self.y, a * self.z) }
    /// Addition.
    #[inline]
    pub fn plus(self, rhs: Self) -> Self { Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z) }
}

impl LieAlgebra for Vec3 {
    type Scalar = f64;
    fn bracket(&self, rhs: &Self) -> Self { self.cross(*rhs) }
    fn scale(&self, scalar: &Self::Scalar) -> Self { (*self).scale(*scalar) }
    fn add(&self, rhs: &Self) -> Self { self.plus(*rhs) }
}

/// Rotation matrix in `SO(3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct So3 {
    matrix: [[f64; 3]; 3],
}

impl So3 {
    /// Identity rotation.
    pub const fn identity() -> Self { Self { matrix: [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] } }
    /// Borrow the rotation matrix.
    pub const fn matrix(&self) -> &[[f64; 3]; 3] { &self.matrix }
    /// Apply the rotation.
    pub fn apply(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.matrix[0][0]*v.x + self.matrix[0][1]*v.y + self.matrix[0][2]*v.z,
            self.matrix[1][0]*v.x + self.matrix[1][1]*v.y + self.matrix[1][2]*v.z,
            self.matrix[2][0]*v.x + self.matrix[2][1]*v.y + self.matrix[2][2]*v.z,
        )
    }
    /// Matrix composition.
    pub fn compose(&self, rhs: &Self) -> Self {
        let mut m = [[0.0;3];3];
        let mut i=0; while i<3 { let mut k=0; while k<3 { let a=self.matrix[i][k]; let mut j=0; while j<3 { m[i][j]+=a*rhs.matrix[k][j]; j+=1; } k+=1; } i+=1; }
        Self { matrix:m }
    }
    /// Exponential map from an axis-angle vector using Rodrigues' formula.
    pub fn exp(omega: Vec3) -> Self {
        let theta = omega.norm();
        if theta < 1e-12 {
            let k = hat(omega);
            let mut m = Self::identity().matrix;
            let mut i=0; while i<3 { let mut j=0; while j<3 { m[i][j]+=k[i][j]; j+=1; } i+=1; }
            return Self { matrix:m };
        }
        let axis = omega.scale(1.0/theta);
        let k = hat(axis);
        let k2 = mat3_mul(k,k);
        let s=theta.sin(); let c=theta.cos();
        let mut m = Self::identity().matrix;
        let mut i=0; while i<3 { let mut j=0; while j<3 { m[i][j]+=s*k[i][j]+(1.0-c)*k2[i][j]; j+=1; } i+=1; }
        Self { matrix:m }
    }
    /// Logarithm map to the principal axis-angle vector.
    pub fn log(&self) -> Vec3 {
        let cos_theta = ((self.matrix[0][0]+self.matrix[1][1]+self.matrix[2][2]-1.0)*0.5).clamp(-1.0,1.0);
        let theta=cos_theta.acos();
        let vee=Vec3::new(self.matrix[2][1]-self.matrix[1][2],self.matrix[0][2]-self.matrix[2][0],self.matrix[1][0]-self.matrix[0][1]);
        if theta < 1e-12 { return vee.scale(0.5); }
        let denom=2.0*theta.sin();
        if denom.abs() < 1e-12 { return vee.scale(0.5); }
        vee.scale(theta/denom)
    }
}

fn hat(v: Vec3)->[[f64;3];3] { [[0.0,-v.z,v.y],[v.z,0.0,-v.x],[-v.y,v.x,0.0]] }
fn mat3_mul(a:[[f64;3];3],b:[[f64;3];3])->[[f64;3];3] { let mut o=[[0.0;3];3]; let mut i=0; while i<3 { let mut k=0; while k<3 { let mut j=0; while j<3 { o[i][j]+=a[i][k]*b[k][j]; j+=1;} k+=1;} i+=1;} o }

/// Twist in `se(3)`, with angular and linear parts.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Se3Tangent {
    /// Angular velocity / axis-angle part.
    pub omega: Vec3,
    /// Translation tangent.
    pub velocity: Vec3,
}

impl LieAlgebra for Se3Tangent {
    type Scalar = f64;
    fn bracket(&self, rhs:&Self)->Self {
        Self { omega:self.omega.cross(rhs.omega), velocity:self.omega.cross(rhs.velocity).plus(self.velocity.cross(rhs.omega)) }
    }
    fn scale(&self,s:&f64)->Self { Self { omega:self.omega.scale(*s), velocity:self.velocity.scale(*s) } }
    fn add(&self,rhs:&Self)->Self { Self { omega:self.omega.plus(rhs.omega), velocity:self.velocity.plus(rhs.velocity) } }
}

/// Rigid transformation in `SE(3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Se3 {
    /// Rotation component.
    pub rotation: So3,
    /// Translation component.
    pub translation: Vec3,
}

impl Se3 {
    /// Identity rigid transform.
    pub const fn identity()->Self { Self { rotation:So3::identity(), translation:Vec3::new(0.0,0.0,0.0) } }
    /// Compose rigid transforms.
    pub fn compose(&self,rhs:&Self)->Self { Self { rotation:self.rotation.compose(&rhs.rotation), translation:self.translation.plus(self.rotation.apply(rhs.translation)) } }
    /// Exponential map from `se(3)` using the SO(3) left Jacobian.
    pub fn exp(xi:Se3Tangent)->Self {
        let theta=xi.omega.norm();
        let r=So3::exp(xi.omega);
        let k=hat(xi.omega);
        let k2=mat3_mul(k,k);
        let (a,b)=if theta<1e-8 { (0.5,1.0/6.0) } else { ((1.0-theta.cos())/(theta*theta),(theta-theta.sin())/(theta*theta*theta)) };
        let v=mat3_add_scaled_identity(k,k2,a,b);
        Self { rotation:r, translation:mat3_vec(v,xi.velocity) }
    }
}
fn mat3_add_scaled_identity(k:[[f64;3];3],k2:[[f64;3];3],a:f64,b:f64)->[[f64;3];3] { let mut o=[[0.0;3];3]; let mut i=0; while i<3 { o[i][i]=1.0; let mut j=0; while j<3 { o[i][j]+=a*k[i][j]+b*k2[i][j]; j+=1;} i+=1;} o }
fn mat3_vec(m:[[f64;3];3],v:Vec3)->Vec3 { Vec3::new(m[0][0]*v.x+m[0][1]*v.y+m[0][2]*v.z,m[1][0]*v.x+m[1][1]*v.y+m[1][2]*v.z,m[2][0]*v.x+m[2][1]*v.y+m[2][2]*v.z) }

/// Second-order Baker-Campbell-Hausdorff approximation
/// `x + y + 1/2[x,y]`.
pub fn bch2<L:LieAlgebra>(x:&L,y:&L)->L {
    let half=L::Scalar::one().checked_div(&L::Scalar::one().add(&L::Scalar::one())).expect("field characteristic must not be two");
    x.add(y).add(&x.bracket(y).scale(&half))
}

/// Fixed-storage Clifford algebra element. `B` is the blade count and must equal
/// `2^N`; `metric[i]` is the square of basis vector `e_i` (`-1`, `0`, or `1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clifford<const N:usize,const B:usize> {
    coeffs:[f64;B],
    metric:[i8;N],
}

impl<const N:usize,const B:usize> Clifford<N,B> {
    /// Construct an element, validating `B == 2^N` and metric entries.
    pub fn new(coeffs:[f64;B],metric:[i8;N])->Option<Self> {
        if N>=usize::BITS as usize || B!=(1usize<<N) || metric.iter().any(|&x| !(-1..=1).contains(&x)) { return None; }
        Some(Self { coeffs,metric })
    }
    /// Scalar element.
    pub fn scalar(value:f64,metric:[i8;N])->Option<Self> { let mut c=[0.0;B]; if B==0{return None;} c[0]=value; Self::new(c,metric) }
    /// Borrow blade coefficients indexed by bit mask.
    pub const fn coefficients(&self)->&[f64;B] { &self.coeffs }
    /// Geometric product.
    pub fn geometric_product(&self,rhs:&Self)->Option<Self> {
        if self.metric!=rhs.metric { return None; }
        let mut out=[0.0;B];
        let mut a=0; while a<B { let mut b=0; while b<B { let (mask,scale)=blade_product::<N>(a,b,&self.metric); out[mask]+=self.coeffs[a]*rhs.coeffs[b]*scale; b+=1;} a+=1; }
        Some(Self { coeffs:out,metric:self.metric })
    }
    /// Reverse anti-automorphism.
    pub fn reverse(&self)->Self {
        let mut out=*self;
        let mut mask=0;
        while mask<B {
            let grade=mask.count_ones();
            if (grade*(grade-1)/2)&1==1 {
                out.coeffs[mask] = -out.coeffs[mask];
            }
            mask+=1;
        }
        out
    }
}
fn blade_product<const N:usize>(a:usize,b:usize,metric:&[i8;N])->(usize,f64) {
    let mut sign=1.0;
    let mut i=0;
    while i<N {
        if (a>>i)&1==1 {
            let lower=b & ((1usize<<i)-1);
            if lower.count_ones()&1==1 {
                sign = -sign;
            }
            if (b>>i)&1==1 {
                sign*=metric[i] as f64;
            }
        }
        i+=1;
    }
    (a^b,sign)
}

/// Unit quaternion representation of `SU(2)`, the double cover of `SO(3)`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Su2 { /// Scalar component.
    pub w:f64, /// First imaginary component.
    pub x:f64, /// Second imaginary component.
    pub y:f64, /// Third imaginary component.
    pub z:f64 }
impl Su2 {
    /// Normalize a quaternion into SU(2).
    pub fn normalized(w:f64,x:f64,y:f64,z:f64)->Option<Self>{ let n=(w*w+x*x+y*y+z*z).sqrt(); if n==0.0{return None;} Some(Self{w:w/n,x:x/n,y:y/n,z:z/n}) }
    /// Group product.
    pub fn compose(self,r:Self)->Self { Self{w:self.w*r.w-self.x*r.x-self.y*r.y-self.z*r.z,x:self.w*r.x+self.x*r.w+self.y*r.z-self.z*r.y,y:self.w*r.y-self.x*r.z+self.y*r.w+self.z*r.x,z:self.w*r.z+self.x*r.y-self.y*r.x+self.z*r.w} }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn so3_exp_log_roundtrip_small(){ let w=Vec3::new(0.2,-0.1,0.3); let got=So3::exp(w).log(); assert!((got.x-w.x).abs()<1e-10); assert!((got.y-w.y).abs()<1e-10); assert!((got.z-w.z).abs()<1e-10); }
    #[test] fn clifford_e1_squared_matches_metric(){ let e1=Clifford::<2,4>::new([0.0,1.0,0.0,0.0],[1,1]).unwrap(); let p=e1.geometric_product(&e1).unwrap(); assert_eq!(p.coefficients()[0],1.0); }
}
