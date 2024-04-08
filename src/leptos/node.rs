use crate::{
    comment::{combine_strs_with_missing_comments, rewrite_missing_comment},
    leptos::convert::{Node, NodeAttribute, Span},
    rewrite::{Rewrite, RewriteContext},
    shape::Shape,
};
use rstml::node::NodeName;
use rustc_ast::MacCall;
use rustc_span::Pos;

use super::convert::{Children, Expr, HasSpan, HasTokens, NodeBlock, NodeElement, NodeFragment};

pub(crate) fn format_leptos_view(
    context: &RewriteContext<'_>,
    shape: Shape,
    mac: &MacCall,
) -> Option<String> {
    let ts = &mac.args.tokens;
    let span = super::convert::ts_span(ts)?;
    let view = context.snippet(span);

    let nested_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);

    let tokens = view.parse::<proc_macro2::TokenStream>().ok()?;
    let nodes = rstml::parse2(tokens.clone()).ok()?;
    let first = Span::from(tokens.into_iter().next()?.span());

    // proc_macro2 parses everything in a fake file, so spans' offsets are always growing,
    // but we need the offset from the beginning of the macro
    let offset = span.lo().to_u32() as isize - first.lo() as isize;

    if let Some(nodes) = Children::from_iter(&nodes, context, ts, offset) {
        let children = rewrite_children(context, nested_shape, &nodes)?;
        // "view! { " + children + " }"
        let do_break = children.len() + 10 > shape.width || children.contains('\n');

        let mut result = combine_strs_with_missing_comments(
            context,
            "view! {",
            &children,
            mac.args.dspan.open.between(span),
            nested_shape,
            !do_break,
        )
        .unwrap_or_default();

        result = combine_strs_with_missing_comments(
            context,
            &result,
            "}",
            span.between(mac.args.dspan.close),
            shape,
            !do_break,
        )
        .unwrap_or_default();

        Some(result)
    } else {
        Some("view! {}".into())
    }
}

pub(crate) fn rewrite_children(
    context: &RewriteContext<'_>,
    shape: Shape,
    children: &Children<'_>,
) -> Option<String> {
    let mut result = String::new();

    let new_children = children
        .inner
        .iter()
        .map(|x| rewrite_node(context, shape, x))
        .collect::<Option<Vec<_>>>()?;

    let children_width =
        new_children.iter().fold(0, |acc, e| acc + e.len()) + new_children.len() - 1;
    let over_width = children_width > shape.width;

    let sep = if over_width {
        shape.indent.to_string_with_newline(context.config)
    } else {
        " ".into()
    };
    result.push_str(&new_children.join(&sep));

    Some(result)
}

pub(crate) fn rewrite_node(
    context: &RewriteContext<'_>,
    shape: Shape,
    node: &Node<'_>,
) -> Option<String> {
    let mut result = String::new();

    match node {
        Node::Comment(x) => result.push_str(&format!("<!-- \"{}\" -->", x.comment.value.value())),
        Node::Doctype(x) => result.push_str(&format!(
            "<!DOCTYPE {}>",
            x.doctype.value.to_token_stream_string()
        )),
        Node::Fragment(x) => result.push_str(&rewrite_fragment(context, shape, x)?),
        Node::Element(x) => result.push_str(&rewrite_element(context, shape, x)?),
        Node::Block(x) => result.push_str(&rewrite_block(context, shape, x)?),
        Node::Text(x) => {
            result.push('"');
            result.push_str(&x.text.value.value());
            result.push('"');
        }
        Node::RawText(x) => result.push_str(&x.raw_text.to_source_text(false)?),
    }

    Some(result)
}

pub(crate) fn rewrite_element(
    context: &RewriteContext<'_>,
    shape: Shape,
    element: &NodeElement<'_>,
) -> Option<String> {
    let name = element.element.name().to_string();
    let is_void = is_void_element(&name, element.children.is_some());
    let opening_tag = rewrite_opening_tag(context, shape, element, is_void)?;

    if is_void {
        return Some(opening_tag);
    }

    let closing_tag = rewrite_closing_tag(element);
    let shape = shape.saturating_sub_width(
        if opening_tag.contains('\n') {
            // '>' token
            1
        } else {
            opening_tag.len()
        } + closing_tag.len(),
    );

    let nested_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);

    if let Some(children) = &element.children {
        let new_children = rewrite_children(context, nested_shape, children)?;

        let span_pre = element.open_tag.span().between(children.span());
        let missing_comment_pre = rewrite_missing_comment(span_pre, shape, context)?;
        let span_post = children.span().between(
            element
                .close_tag
                .as_ref()
                .map(|x| x.span())
                .unwrap_or_else(|| children.span()),
        );
        let missing_comment_post = rewrite_missing_comment(span_post, shape, context)?;
        if !missing_comment_pre.is_empty()
            || !missing_comment_post.is_empty()
            || opening_tag.len() + new_children.len() + closing_tag.len() > shape.width
        {
            let mut result = combine_strs_with_missing_comments(
                context,
                &opening_tag,
                &new_children,
                element.open_tag.span().between(children.span()),
                nested_shape,
                false,
            )?;

            result = combine_strs_with_missing_comments(
                context,
                &result,
                &closing_tag,
                children.span().between(
                    element
                        .close_tag
                        .as_ref()
                        .map(|x| x.span())
                        .unwrap_or_else(|| children.span()),
                ),
                shape,
                false,
            )?;

            Some(result)
        } else {
            Some(format!("{opening_tag}{new_children}{closing_tag}"))
        }
    } else {
        Some(format!("{opening_tag}{closing_tag}"))
    }
}

pub(crate) fn rewrite_block(
    context: &RewriteContext<'_>,
    shape: Shape,
    block: &NodeBlock<'_>,
) -> Option<String> {
    let ast = match &block.ast.kind {
        rustc_ast::ExprKind::Block(x, _) => x,
        _ => unreachable!(),
    };

    Some(match ast.stmts.first() {
        Some(rustc_ast::Stmt {
            kind: rustc_ast::StmtKind::Expr(x),
            ..
        }) if ast.stmts.len() == 1 => format!("{{{}}}", x.rewrite(context, shape)?),
        _ => ast.rewrite(context, shape)?,
    })
}

pub(crate) fn rewrite_fragment(
    context: &RewriteContext<'_>,
    shape: Shape,
    frag: &NodeFragment<'_>,
) -> Option<String> {
    let mut result: String;

    if let Some(children) = frag.children.as_ref() {
        let nested_shape = shape
            .saturating_sub_width(5)
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config);

        let new_children = rewrite_children(context, nested_shape, children)?;

        let add_opening_frag = |shape| {
            combine_strs_with_missing_comments(
                context,
                "<>",
                &new_children,
                frag.open_fragment.span().between(children.span()),
                shape,
                false,
            )
        };

        result = add_opening_frag(nested_shape)
            .or_else(|| add_opening_frag(nested_shape.infinite_width()))?;

        let add_closing_frag = |shape| {
            combine_strs_with_missing_comments(
                context,
                &result,
                "</>",
                children
                    .span()
                    .between(frag.close_fragment.as_ref()?.span()),
                shape,
                false,
            )
        };
        result = add_closing_frag(shape).or_else(|| add_closing_frag(shape.infinite_width()))?;
        Some(result)
    } else {
        result = combine_strs_with_missing_comments(
            context,
            "<>",
            "</>",
            frag.open_fragment
                .span()
                .between(frag.close_fragment.as_ref()?.span()),
            shape,
            false,
        )?;
        Some(result)
    }
}

pub(crate) fn rewrite_opening_tag(
    context: &RewriteContext<'_>,
    shape: Shape,
    element: &NodeElement<'_>,
    is_void: bool,
) -> Option<String> {
    let mut result = String::new();

    result.push('<');
    result.push_str(&rewrite_node_name(&element.element.open_tag.name));

    let shape = shape.saturating_sub_width(result.len() + if is_void { 2 } else { 1 });
    result.push_str(&rewrite_attributes(context, shape, &element.attributes)?);

    if is_void {
        result.push_str("/>");
    } else {
        result.push('>');
    }

    Some(result)
}

pub(crate) fn rewrite_closing_tag(element: &NodeElement<'_>) -> String {
    format!("</{}>", rewrite_node_name(element.element.name()))
}

pub(crate) fn rewrite_attributes(
    context: &RewriteContext<'_>,
    shape: Shape,
    attributes: &[NodeAttribute<'_>],
) -> Option<String> {
    let mut result = String::new();

    if attributes.is_empty() {
        return Some(result);
    }

    let nested_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);

    let new_attributes = attributes
        .iter()
        .map(|x| rewrite_attribute(context, nested_shape, x))
        .collect::<Option<Vec<_>>>()?;
    let new_width = new_attributes.iter().fold(0, |acc, e| acc + e.len() + 1);
    let do_break = new_width > shape.width || new_attributes.iter().any(|x| x.contains('\n'));
    let sep = if do_break {
        nested_shape.to_string_with_newline(context.config)
    } else {
        " ".into()
    };

    if let [attr] = &new_attributes[..] {
        result.push(' ');
        result.push_str(attr);
    } else {
        result.push_str(&sep);
        result.push_str(&new_attributes.join(&sep));
    }
    if do_break {
        result.push_str(&shape.to_string_with_newline(context.config));
    }

    Some(result)
}

pub(crate) fn rewrite_attribute(
    context: &RewriteContext<'_>,
    shape: Shape,
    attr: &NodeAttribute<'_>,
) -> Option<String> {
    match attr {
        NodeAttribute::Block(x) => rewrite_block(context, shape, x),
        NodeAttribute::Attribute(x) => {
            let name = rewrite_node_name(&x.keyed_attribute.key);

            if let Some(x) = &x.value {
                Some(format!(
                    "{name}={}",
                    rewrite_expr(context, shape.saturating_sub_width(name.len() + 1), x)?
                ))
            } else {
                Some(name)
            }
        }
    }
}

pub(crate) fn rewrite_expr(
    context: &RewriteContext<'_>,
    shape: Shape,
    expr: &Expr<'_>,
) -> Option<String> {
    let mut parser =
        rustc_parse::stream_to_parser(context.parse_sess.inner(), expr.tokens().clone(), None);
    let expr = parser.parse_expr().ok()?;

    expr.rewrite(context, shape)
}

pub(crate) fn rewrite_node_name(name: &NodeName) -> String {
    name.to_string()
}

fn is_void_element(name: &str, has_children: bool) -> bool {
    if name.chars().next().unwrap().is_uppercase() {
        !has_children
    } else {
        matches!(
            name,
            "area"
                | "base"
                | "br"
                | "col"
                | "embed"
                | "hr"
                | "img"
                | "input"
                | "link"
                | "meta"
                | "param"
                | "source"
                | "track"
                | "wbr"
        )
    }
}
