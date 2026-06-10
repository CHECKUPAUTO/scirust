use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::visit_mut::VisitMut;
use syn::{Expr, ExprLit, FnArg, Ident, ItemFn, Lit, Pat, PatType, Type, parse_macro_input};

/// Visitor that replaces argument identifiers with their `_dual` counterpart.
struct ArgReplacer {
    arg_map: std::collections::HashMap<Ident, Ident>,
}

impl VisitMut for ArgReplacer {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Path(expr_path) = expr
        {
            if expr_path.path.leading_colon.is_none() && expr_path.path.segments.len() == 1
            {
                let ident = &expr_path.path.segments[0].ident;
                if let Some(new_ident) = self.arg_map.get(ident)
                {
                    expr_path.path = syn::parse_quote!(#new_ident);
                    return;
                }
            }
        }
        // Replace float literals with Dual::primal(literal)
        if let Expr::Lit(ExprLit {
            lit: Lit::Float(lit_float),
            ..
        }) = expr
        {
            if let Ok(val) = lit_float.base10_parse::<f64>()
            {
                *expr = syn::parse_quote! {
                    scirust_autodiff::Dual::primal(#val)
                };
                return;
            }
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }
}

#[proc_macro_attribute]
pub fn autodiff(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input_fn = parse_macro_input!(item as ItemFn);

    // Remove #[autodiff] from the input function to avoid infinite recursion
    input_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("autodiff"));

    let sig = &input_fn.sig;
    let fn_name = &sig.ident;

    // Collect argument info: (ident, type_is_f64)
    let mut arg_names = Vec::new();
    for arg in &sig.inputs
    {
        match arg
        {
            FnArg::Typed(PatType { pat, ty, .. }) =>
            {
                if let Pat::Ident(pat_ident) = pat.as_ref()
                {
                    let is_f64 = if let Type::Path(type_path) = ty.as_ref()
                    {
                        type_path.path.is_ident("f64")
                    }
                    else
                    {
                        false
                    };
                    if !is_f64
                    {
                        return syn::Error::new_spanned(
                            ty,
                            "autodiff currently only supports f64 arguments",
                        )
                        .to_compile_error()
                        .into();
                    }
                    arg_names.push(pat_ident.ident.clone());
                }
                else
                {
                    return syn::Error::new_spanned(
                        pat,
                        "autodiff only supports simple identifier arguments",
                    )
                    .to_compile_error()
                    .into();
                }
            },
            FnArg::Receiver(_) =>
            {
                return syn::Error::new_spanned(arg, "autodiff does not support `self` parameters")
                    .to_compile_error()
                    .into();
            },
        }
    }

    if arg_names.is_empty()
    {
        return syn::Error::new_spanned(&sig.ident, "autodiff requires at least one argument")
            .to_compile_error()
            .into();
    }

    let vis = &input_fn.vis;
    let grad_fn_name = format_ident!("{}_grad", fn_name);
    let dual_names: Vec<_> = arg_names
        .iter()
        .map(|name| format_ident!("{}_dual", name))
        .collect();

    let mut arg_map = std::collections::HashMap::new();
    for (name, dual) in arg_names.iter().zip(dual_names.iter())
    {
        arg_map.insert(name.clone(), dual.clone());
    }

    let mut transformed_bodies = Vec::new();
    for i in 0..arg_names.len()
    {
        let mut replacer = ArgReplacer {
            arg_map: arg_map.clone(),
        };
        let mut body = input_fn.block.clone();
        replacer.visit_block_mut(&mut body);

        let dual_inits = dual_names.iter().enumerate().map(|(j, dname)| {
            let aname = &arg_names[j];
            if i == j
            {
                quote! { let #dname = scirust_autodiff::Dual::var(#aname); }
            }
            else
            {
                quote! { let #dname = scirust_autodiff::Dual::primal(#aname); }
            }
        });

        transformed_bodies.push(quote! {
            {
                #(#dual_inits)*
                let result: scirust_autodiff::Dual = #body;
                result.grad()
            }
        });
    }

    let inputs = &sig.inputs;
    let return_type = if arg_names.len() == 1
    {
        quote! { f64 }
    }
    else
    {
        let types = vec![quote! { f64 }; arg_names.len()];
        quote! { (#(#types),*) }
    };

    let return_value = if arg_names.len() == 1
    {
        quote! { #(#transformed_bodies)* }
    }
    else
    {
        quote! { (#(#transformed_bodies),*) }
    };

    let expanded = quote! {
        #input_fn

        #vis fn #grad_fn_name(#inputs) -> #return_type {
            #return_value
        }
    };

    TokenStream::from(expanded)
}
