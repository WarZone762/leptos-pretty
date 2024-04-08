use proc_macro2 as pm;
use rstml::{atoms, node};
use rustc_ast::tokenstream::TokenStream;
use rustc_span::{self as rc, Pos};

use crate::rewrite::RewriteContext;

pub(crate) fn corresponding_ts(
    ts: &TokenStream,
    src: impl quote::ToTokens,
    offset: isize,
) -> TokenStream {
    let src_tokens = src.to_token_stream();

    let mut iter = src_tokens.into_iter();
    let first = Span::from(iter.next().unwrap().span());

    let start = (first.lo() as isize + offset) as u32;
    let end = (iter
        .last()
        .map(|x| Span::from(x.span()).hi())
        .unwrap_or(first.hi()) as isize
        + offset) as u32;

    ts.trees()
        .skip_while(move |x| x.span().hi().to_u32() <= start)
        .take_while(move |x| x.span().lo().to_u32() < end)
        .cloned()
        .collect()
}

pub(crate) fn ts_span(ts: &TokenStream) -> Option<rc::Span> {
    let mut iter = ts.trees();
    let first = iter.next()?.span();
    let last = iter.last().map(|x| x.span()).unwrap_or(first);

    Some(first.to(last))
}

fn ts_and_span(
    ts: &TokenStream,
    src: impl quote::ToTokens,
    offset: isize,
) -> (TokenStream, rc::Span) {
    let ts = corresponding_ts(ts, src, offset);
    let span = ts_span(&ts).expect("unable to get span of empty token stream");
    (ts, span)
}

pub(crate) trait HasTokens {
    fn tokens(&self) -> &TokenStream;
}

pub(crate) trait HasSpan {
    fn span(&self) -> rc::Span;
}

macro_rules! impl_span_tokens {
    ($struct:ty) => {
        impl HasTokens for $struct {
            fn tokens(&self) -> &TokenStream {
                &self.tokens
            }
        }

        impl HasSpan for $struct {
            fn span(&self) -> rc::Span {
                self.span
            }
        }
    };
}

#[derive(Debug, Clone)]
pub(crate) enum Node<'a> {
    Comment(NodeComment<'a>),
    Doctype(NodeDoctype<'a>),
    Fragment(NodeFragment<'a>),
    Element(NodeElement<'a>),
    Block(NodeBlock<'a>),
    Text(NodeText<'a>),
    RawText(NodeRawText<'a>),
}

impl HasTokens for Node<'_> {
    fn tokens(&self) -> &TokenStream {
        match self {
            Node::Comment(x) => x.tokens(),
            Node::Doctype(x) => x.tokens(),
            Node::Fragment(x) => x.tokens(),
            Node::Element(x) => x.tokens(),
            Node::Block(x) => x.tokens(),
            Node::Text(x) => x.tokens(),
            Node::RawText(x) => x.tokens(),
        }
    }
}

impl HasSpan for Node<'_> {
    fn span(&self) -> rc::Span {
        match self {
            Node::Comment(x) => x.span(),
            Node::Doctype(x) => x.span(),
            Node::Fragment(x) => x.span(),
            Node::Element(x) => x.span(),
            Node::Block(x) => x.span(),
            Node::Text(x) => x.span(),
            Node::RawText(x) => x.span(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct NodeComment<'a> {
    pub(crate) comment: &'a node::NodeComment,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeComment<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeDoctype<'a> {
    pub(crate) doctype: &'a node::NodeDoctype,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeDoctype<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeFragment<'a> {
    pub(crate) open_fragment: FragmentOpen<'a>,
    pub(crate) close_fragment: Option<FragmentClose<'a>>,
    pub(crate) children: Option<Children<'a>>,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeFragment<'_>);

#[derive(Debug, Clone)]
pub(crate) struct Children<'a> {
    /// Never empty
    pub(crate) inner: Vec<Node<'a>>,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(Children<'_>);

impl<'a> Children<'a> {
    /// Returns None if `ts` is empty
    pub(crate) fn from_iter<T: IntoIterator<Item = &'a node::Node>>(
        iter: T,
        context: &RewriteContext<'_>,
        ts: &TokenStream,
        offset: isize,
    ) -> Option<Self> {
        let mut tokens = TokenStream::default();

        let children = iter
            .into_iter()
            .map(|node| {
                tokens.push_stream(corresponding_ts(ts, node, offset));
                Node::from_node(context, node, ts, offset)
            })
            .collect();

        let span = ts_span(&tokens)?;
        Some(Self {
            inner: children,
            tokens,
            span,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FragmentOpen<'a> {
    pub(crate) open_fragment: &'a atoms::FragmentOpen,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(FragmentOpen<'_>);

#[derive(Debug, Clone)]
pub(crate) struct FragmentClose<'a> {
    pub(crate) close_fragment: &'a atoms::FragmentClose,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(FragmentClose<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeElement<'a> {
    pub(crate) element: &'a node::NodeElement,
    pub(crate) open_tag: OpenTag<'a>,
    pub(crate) close_tag: Option<CloseTag<'a>>,
    pub(crate) attributes: Vec<NodeAttribute<'a>>,
    pub(crate) children: Option<Children<'a>>,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeElement<'_>);

#[derive(Debug, Clone)]
pub(crate) struct OpenTag<'a> {
    pub(crate) open_tag: &'a atoms::OpenTag,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(OpenTag<'_>);

#[derive(Debug, Clone)]
pub(crate) struct CloseTag<'a> {
    pub(crate) close_tag: &'a atoms::CloseTag,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(CloseTag<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeBlock<'a> {
    pub(crate) block: &'a node::NodeBlock,
    pub(crate) ast: rustc_ast::ptr::P<rustc_ast::Expr>,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeBlock<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeText<'a> {
    pub(crate) text: &'a node::NodeText,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeText<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeRawText<'a> {
    pub(crate) raw_text: &'a node::RawText,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(NodeRawText<'_>);

#[derive(Debug, Clone)]
pub(crate) enum NodeAttribute<'a> {
    Block(NodeBlock<'a>),
    Attribute(KeyedAttribute<'a>),
}

impl NodeAttribute<'_> {
    pub(crate) fn from_node<'a>(
        context: &RewriteContext<'_>,
        attr: &'a node::NodeAttribute,
        ts: &TokenStream,
        offset: isize,
    ) -> NodeAttribute<'a> {
        match attr {
            node::NodeAttribute::Block(x) => {
                let (tokens, span) = ts_and_span(ts, x, offset);

                let mut parser =
                    rustc_parse::stream_to_parser(context.parse_sess.inner(), tokens.clone(), None);

                NodeAttribute::Block(NodeBlock {
                    block: x,
                    ast: parser.parse_expr().unwrap(),
                    tokens,
                    span,
                })
            }
            node::NodeAttribute::Attribute(x) => {
                let (tokens, span) = ts_and_span(ts, x, offset);
                NodeAttribute::Attribute(KeyedAttribute {
                    keyed_attribute: x,
                    value: x.value().map(|expr| {
                        let (tokens, span) = ts_and_span(ts, expr, offset);
                        Expr { expr, tokens, span }
                    }),
                    tokens,
                    span,
                })
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct KeyedAttribute<'a> {
    pub(crate) keyed_attribute: &'a node::KeyedAttribute,
    pub(crate) value: Option<Expr<'a>>,
    tokens: TokenStream,
    span: rc::Span,
}

#[derive(Debug, Clone)]
pub(crate) struct Expr<'a> {
    pub(crate) expr: &'a syn::Expr,
    tokens: TokenStream,
    span: rc::Span,
}

impl_span_tokens!(Expr<'_>);

impl Node<'_> {
    pub(crate) fn from_node<'a>(
        context: &RewriteContext<'_>,
        node: &'a node::Node,
        ts: &TokenStream,
        offset: isize,
    ) -> Node<'a> {
        let (tokens, span) = ts_and_span(ts, node, offset);

        match node {
            node::Node::Fragment(x) => Node::Fragment(NodeFragment {
                open_fragment: {
                    let (tokens, span) = ts_and_span(ts, &x.tag_open, offset);
                    FragmentOpen {
                        open_fragment: &x.tag_open,
                        tokens,
                        span,
                    }
                },
                close_fragment: x.tag_close.as_ref().map(|close_fragment| {
                    let (tokens, span) = ts_and_span(ts, close_fragment, offset);
                    FragmentClose {
                        close_fragment,
                        tokens,
                        span,
                    }
                }),
                children: Children::from_iter(&x.children, context, ts, offset),
                tokens,
                span,
            }),
            node::Node::Element(x) => Node::Element(NodeElement {
                element: x,
                open_tag: {
                    let (tokens, span) = ts_and_span(ts, &x.open_tag, offset);
                    OpenTag {
                        open_tag: &x.open_tag,
                        tokens,
                        span,
                    }
                },
                close_tag: x.close_tag.as_ref().map(|close_tag| {
                    let (tokens, span) = ts_and_span(ts, close_tag, offset);
                    CloseTag {
                        close_tag,
                        tokens,
                        span,
                    }
                }),
                attributes: x
                    .attributes()
                    .iter()
                    .map(|attr| NodeAttribute::from_node(context, attr, ts, offset))
                    .collect(),
                children: Children::from_iter(&x.children, context, ts, offset),
                tokens,
                span,
            }),
            node::Node::Doctype(x) => Node::Doctype(NodeDoctype {
                doctype: x,
                tokens,
                span,
            }),
            node::Node::Comment(x) => Node::Comment(NodeComment {
                comment: x,
                tokens,
                span,
            }),
            node::Node::Block(x) => {
                let mut parser =
                    rustc_parse::stream_to_parser(context.parse_sess.inner(), tokens.clone(), None);
                Node::Block(NodeBlock {
                    block: x,
                    ast: parser.parse_expr().unwrap(),
                    tokens,
                    span,
                })
            }
            node::Node::Text(x) => Node::Text(NodeText {
                text: x,
                tokens,
                span,
            }),
            node::Node::RawText(x) => Node::RawText(NodeRawText {
                raw_text: x,
                tokens,
                span,
            }),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct Span {
    pub(crate) lo: u32,
    pub(crate) hi: u32,
}

impl Span {
    pub(crate) fn lo(&self) -> u32 {
        self.lo
    }

    pub(crate) fn hi(&self) -> u32 {
        self.hi
    }
}

impl From<pm::Span> for Span {
    fn from(value: pm::Span) -> Self {
        let range = value.byte_range();
        Self {
            lo: range.start as u32,
            hi: range.end as u32,
        }
    }
}
