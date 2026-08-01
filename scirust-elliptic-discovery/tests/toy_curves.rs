use scirust_elliptic_discovery::{CurveError, FieldError, Fp, PrimeError, ToyCurve, ToyPrime};

fn curve_over_17() -> ToyCurve {
    ToyCurve::new(ToyPrime::new(17).expect("17 is a toy prime"), 2, 2)
        .expect("curve is nonsingular")
}

#[test]
fn toy_prime_rejects_outside_or_composite_values() {
    assert_eq!(ToyPrime::new(3), Err(PrimeError::BelowMinimum { value: 3 }));
    assert_eq!(
        ToyPrime::new(4094),
        Err(PrimeError::AboveMaximum { value: 4094 })
    );
    assert_eq!(ToyPrime::new(21), Err(PrimeError::Composite { value: 21 }));
}

#[test]
fn field_operations_are_exact_and_field_checked() {
    let seventeen = ToyPrime::new(17).expect("17 is prime");
    let nineteen = ToyPrime::new(19).expect("19 is prime");
    let fourteen = Fp::new(seventeen, 14);

    assert_eq!(
        fourteen.checked_add(Fp::new(seventeen, 8)),
        Ok(Fp::new(seventeen, 5))
    );
    assert_eq!(
        fourteen.checked_mul(Fp::new(seventeen, 3)),
        Ok(Fp::new(seventeen, 8))
    );
    assert_eq!(
        fourteen.checked_div(Fp::new(seventeen, 2)),
        Ok(Fp::new(seventeen, 7))
    );
    assert_eq!(
        fourteen.checked_add(Fp::new(nineteen, 1)),
        Err(FieldError::DifferentPrimes)
    );
    assert_eq!(Fp::new(seventeen, 0).inverse(), None);
}

#[test]
fn singular_curve_and_invalid_point_are_rejected() {
    let five = ToyPrime::new(5).expect("5 is prime");
    assert_eq!(ToyCurve::new(five, 0, 0), Err(CurveError::Singular));

    let curve = curve_over_17();
    assert_eq!(
        curve.point_from_local_residues(5, 2),
        Err(CurveError::PointNotOnCurve)
    );
    assert_eq!(
        curve.point_from_local_residues(17, 1),
        Err(CurveError::CoordinateOutsideField)
    );
}

#[test]
fn group_law_matches_known_exact_example() {
    let curve = curve_over_17();
    let point = curve
        .point_from_local_residues(5, 1)
        .expect("point is on the curve");
    let other = curve
        .point_from_local_residues(6, 3)
        .expect("point is on the curve");

    assert_eq!(
        curve
            .add(point, other)
            .expect("addition succeeds")
            .affine_coordinates(),
        Some((10, 6))
    );
    assert_eq!(
        curve
            .add(point, point)
            .expect("doubling succeeds")
            .affine_coordinates(),
        Some((6, 3))
    );
    assert!(
        curve
            .add(point, curve.negate(point).expect("negation succeeds"))
            .expect("addition succeeds")
            .is_infinity()
    );
}

#[test]
fn enumeration_orders_and_hasse_bound_are_exact() {
    let curve = curve_over_17();
    let points = curve.enumerate_points();

    assert_eq!(points.len(), 19);
    assert!(points[0].is_infinity());
    assert_eq!(points[1].affine_coordinates(), Some((0, 6)));
    assert!(points.iter().all(|point| curve.is_on_curve(point)));
    assert!(curve.satisfies_hasse_bound());

    let generator = curve
        .point_from_local_residues(5, 1)
        .expect("point is on the curve");
    assert_eq!(curve.group_order(), 19);
    assert_eq!(curve.point_order(generator), Ok(19));
    assert!(
        curve
            .scalar_mul(generator, 19)
            .expect("scalar multiplication succeeds")
            .is_infinity()
    );
}

#[test]
fn every_enumerated_sum_stays_on_its_curve() {
    let curve = curve_over_17();
    let points = curve.enumerate_points();

    for left in &points
    {
        for right in &points
        {
            let sum = curve
                .add(*left, *right)
                .expect("enumerated points are valid");
            assert!(curve.is_on_curve(&sum));
        }
    }
}

fn repeated_order(curve: ToyCurve, point: scirust_elliptic_discovery::ToyPoint) -> u64 {
    let mut multiple = curve.identity();
    for scalar in 1..=curve.group_order()
    {
        multiple = curve
            .add(multiple, point)
            .expect("enumerated points stay on their curve");
        if multiple.is_infinity()
        {
            return scalar;
        }
    }
    panic!("a finite group point must reach the identity within the group order");
}

#[test]
fn exhaustive_small_corpus_validates_group_law_and_point_orders() {
    let mut nonsingular_curve_count = 0u64;

    for modulus in [5u64, 7, 11, 13]
    {
        let prime = ToyPrime::new(modulus).expect("listed modulus is a toy prime");
        for a in 0..modulus
        {
            for b in 0..modulus
            {
                let curve = match ToyCurve::new(prime, a, b)
                {
                    Ok(curve) => curve,
                    Err(CurveError::Singular) => continue,
                    Err(error) => panic!("unexpected curve construction error: {error}"),
                };
                nonsingular_curve_count += 1;

                let points = curve.enumerate_points();
                assert_eq!(points, curve.enumerate_points());
                assert!(curve.satisfies_hasse_bound());

                for point in &points
                {
                    assert!(curve.is_on_curve(point));
                    assert_eq!(
                        curve
                            .add(*point, curve.identity())
                            .expect("identity addition succeeds"),
                        *point
                    );
                    assert!(
                        curve
                            .add(*point, curve.negate(*point).expect("negation succeeds"))
                            .expect("inverse addition succeeds")
                            .is_infinity()
                    );
                    assert!(
                        curve
                            .scalar_mul(*point, curve.group_order())
                            .expect("scalar multiplication succeeds")
                            .is_infinity()
                    );

                    let computed_order = curve.point_order(*point).expect("point is valid");
                    let independent_order = repeated_order(curve, *point);
                    assert_eq!(computed_order, independent_order);
                    assert_eq!(curve.group_order() % computed_order, 0);
                }

                for left in &points
                {
                    for right in &points
                    {
                        let sum = curve
                            .add(*left, *right)
                            .expect("enumerated points are valid");
                        assert!(curve.is_on_curve(&sum));
                    }
                }
            }
        }
    }

    assert!(nonsingular_curve_count > 0);
}

#[test]
fn operations_reject_points_from_another_curve() {
    let seventeen = ToyPrime::new(17).expect("17 is prime");
    let left = ToyCurve::new(seventeen, 2, 2).expect("curve is nonsingular");
    let right = ToyCurve::new(seventeen, 2, 3).expect("curve is nonsingular");

    assert_eq!(
        left.add(left.identity(), right.identity()),
        Err(CurveError::PointFromAnotherCurve)
    );
}
