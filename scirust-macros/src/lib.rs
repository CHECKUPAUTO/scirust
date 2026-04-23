use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::visit_mut::VisitMut;
use syn::{parse_macro_input, Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Pat, PatType, Type};

/// Visitor that replaces argument identifiers with their `_dual` counterpart.
struct ArgReplacer {
    arg_map: std::collections::HashMap<Ident, Ident>,
}

impl VisitMut for ArgReplacer {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Path(expr_path) = expr {
            if expr_path.path.get_ident().is_some() {
                let ident = expr_path.path.get_ident().unwrap();
                if let Some(new_ident) = self.arg_map.get(ident) {
                    expr_path.path = syn::parse_quote!(#new_ident);
                    return;
                }
            }
        }
        // Replace float literals with Dual::primal(literal)
        if let Expr::Lit(ExprLit { lit: Lit::Float(lit_float), .. }) = expr {
            let val = &lit_float.base10_parse::<f64>().unwrap();
            let lit_str = format!("{:.}", val);
            *expr = syn::parse_quote! {
                scirust_autodiff::Dual::primal(#lit_str.parse::<f64>().unwrap())
            };
            return;
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }
}

#[proc_macro_attribute]
pub fn autodiff(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;

    // Collect argument info: (ident, type_is_f64)
    let mut arg_info: Vec<(Ident, bool)> = Vec::new();
    for arg in &sig.inputs {
        match arg {
            FnArg::Typed(PatType { pat, ty, .. }) => {
                if let Pat::Ident(pat_ident) = pat.as_ref() {
                    let is_f64 = if let Type::Path(type_path) = ty.as_ref() {
                        type_path.path.is_ident("f64")
                    } else {
                        false
                    };
                    arg_info.push((pat_ident.ident.clone(), is_f64));
                } else {
                    panic!("autodiff only supports simple identifier arguments");
                }
            }
            FnArg::Receiver(_) => {
                panic!("autodiff does not support `self` parameters");
            }
        }
    }

    let num_args = arg_info.len();
    if num_args == 0 || num_args > 4 {
        panic!("autodiff supports functions with 1 to 4 f64 arguments");
    }

    // Check all args are f64
    for (_, is_f64) in &arg_info {
        if !is_f64 {
            panic!("autodiff currently only supports f64 arguments");
        }
    }

    let vis = &input_fn.vis;
    let grad_fn_name = format_ident!("{}_grad", fn_name);

    match num_args {
        1 => {
            let arg_name = &arg_info[0].0;
            let dual_name = format_ident!("{}_dual", arg_name);

            let mut arg_map = std::collections::HashMap::new();
            arg_map.insert(arg_name.clone(), dual_name.clone());

            let mut replacer = ArgReplacer { arg_map };
            let mut body = input_fn.block.clone();
            replacer.visit_block_mut(&mut body);

            let expanded = quote! {
                #input_fn

                #vis fn #grad_fn_name(#arg_name: f64) -> f64 {
                    use scirust_autodiff::Dual;
                    let #dual_name = Dual::var(#arg_name);
                    let result = #body;
                    result.grad()
                }
            };

            TokenStream::from(expanded)
        }
        2 => {
            let arg1 = &arg_info[0].0;
            let arg2 = &arg_info[1].0;
            let dual1 = format_ident!("{}_dual", arg1);
            let dual2 = format_ident!("{}_dual", arg2);

            // dx: arg1 is active, arg2 is passive
            let mut arg_map_dx = std::collections::HashMap::new();
            arg_map_dx.insert(arg1.clone(), dual1.clone());
            arg_map_dx.insert(arg2.clone(), dual2.clone());
            let mut replacer_dx = ArgReplacer { arg_map: arg_map_dx };
            let mut body_dx = input_fn.block.clone();
            replacer_dx.visit_block_mut(&mut body_dx);

            // dy: arg1 is passive, arg2 is active
            let mut arg_map_dy = std::collections::HashMap::new();
            arg_map_dy.insert(arg1.clone(), dual1.clone());
            arg_map_dy.insert(arg2.clone(), dual2.clone());
            let mut replacer_dy = ArgReplacer { arg_map: arg_map_dy };
            let mut body_dy = input_fn.block.clone();
            replacer_dy.visit_block_mut(&mut body_dy);

            let expanded = quote! {
                #input_fn

                #vis fn #grad_fn_name(#arg1: f64, #arg2: f64) -> (f64, f64) {
                    use scirust_autodiff::Dual;
                    let #dual1 = Dual::var(#arg1);
                    let #dual2 = Dual::primal(#arg2);
                    let result_dx = #body_dx;
                    let #dual1 = Dual::primal(#arg1);
                    let #dual2 = Dual::var(#arg2);
                    let result_dy = #body_dy;
                    (result_dx.grad(), result_dy.grad())
                }
            };

            TokenStream::from(expanded)
        }
        3 => {
            let a1 = &arg_info[0].0;
            let a2 = &arg_info[1].0;
            let a3 = &arg_info[2].0;
            let d1 = format_ident!("{}_dual", a1);
            let d2 = format_ident!("{}_dual", a2);
            let d3 = format_ident!("{}_dual", a3);

            let mut arg_map = std::collections::HashMap::new();
            arg_map.insert(a1.clone(), d1.clone());
            arg_map.insert(a2.clone(), d2.clone());
            arg_map.insert(a3.clone(), d3.clone());

            let make_body = |active: usize| {
                let mut replacer = ArgReplacer { arg_map: arg_map.clone() };
                let mut body = input_fn.block.clone();
                replacer.visit_block_mut(&mut body);
                body
            };

            let body_dx = make_body(0);
            let body_dy = make_body(1);
            let body_dz = make_body(2);

            let expanded = quote! {
                #input_fn

                #vis fn #grad_fn_name(#a1: f64, #a2: f64, #a3: f64) -> (f64, f64, f64) {
                    use scirust_autodiff::Dual;
                    let #d1 = Dual::var(#a1); let #d2 = Dual::primal(#a2); let #d3 = Dual::primal(#a3);
                    let result_dx = #body_dx;
                    let #d1 = Dual::primal(#a1); let #d2 = Dual::var(#a2); let #d3 = Dual::primal(#a3);
                    let result_dy = #body_dy;
                    let #d1 = Dual::primal(#a1); let #d2 = Dual::primal(#a2); let #d3 = Dual::var(#a3);
                    let result_dz = #body_dz;
                    (result_dx.grad(), result_dy.grad(), result_dz.grad())
                }
            };

            TokenStream::from(expanded)
        }
        4 => {
            let a1 = &arg_info[0].0;
            let a2 = &arg_info[1].0;
            let a3 = &arg_info[2].0;
            let a4 = &arg_info[3].0;
            let d1 = format_ident!("{}_dual", a1);
            let d2 = format_ident!("{}_dual", a2);
            let d3 = format_ident!("{}_dual", a3);
            let d4 = format_ident!("{}_dual", a4);

            let mut arg_map = std::collections::HashMap::new();
            arg_map.insert(a1.clone(), d1.clone());
            arg_map.insert(a2.clone(), d2.clone());
            arg_map.insert(a3.clone(), d3.clone());
            arg_map.insert(a4.clone(), d4.clone());

            let make_body = || {
                let mut replacer = ArgReplacer { arg_map: arg_map.clone() };
                let mut body = input_fn.block.clone();
                replacer.visit_block_mut(&mut body);
                body
            };

            let b_dx = make_body();
            let b_dy = make_body();
            let b_dz = make_body();
            let b_dw = make_body();

            let expanded = quote! {
                #input_fn

                #vis fn #grad_fn_name(#a1: f64, #a2: f64, #a3: f64, #a4: f64) -> (f64, f64, f64, f64) {
                    use scirust_autodiff::Dual;
                    let #d1 = Dual::var(#a1); let #d2 = Dual::primal(#a2); let #d3 = Dual::primal(#a3); let #d4 = Dual::primal(#a4);
                    let r_dx = #b_dx;
                    let #d1 = Dual::primal(#a1); let #d2 = Dual::var(#a2); let #d3 = Dual::primal(#a3); let #d4 = Dual::primal(#a4);
                    let r_dy = #b_dy;
                    let #d1 = Dual::primal(#a1); let #d2 = Dual::primal(#a2); let #d3 = Dual::var(#a3); let #d4 = Dual::primal(#a4);
                    let r_dz = #b_dz;
                    let #d1 = Dual::primal(#a1); let #d2 = Dual::primal(#a2); let #d3 = Dual::primal(#a3); let #d4 = Dual::var(#a4);
                    let r_dw = #b_dw;
                    (r_dx.grad(), r_dy.grad(), r_dz.grad(), r_dw.grad())
                }
            };

            TokenStream::from(expanded)
        }
        _ => unreachable!(),
    }
}
