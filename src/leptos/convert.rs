use std::mem;

use proc_macro2 as pm;
use rstml::{atoms, node};
use rustc_ast::tokenstream::TokenStream;
use rustc_span::{Pos, Span};

use crate::rewrite::RewriteContext;

pub(crate) fn corresponding_ts(
    ts: &TokenStream,
    src: impl quote::ToTokens,
    offset: isize,
) -> TokenStream {
    let src_tokens = src.to_token_stream();

    let mut iter = src_tokens.into_iter();
    let first = _Span::from(iter.next().unwrap().span());

    let start = (first.lo() as isize + offset) as u32;
    let end = (iter
        .last()
        .map(|x| _Span::from(x.span()).hi())
        .unwrap_or(first.hi()) as isize
        + offset) as u32;

    ts.trees()
        .skip_while(move |x| x.span().hi().to_u32() <= start)
        .take_while(move |x| x.span().lo().to_u32() < end)
        .cloned()
        .collect()
}

pub(crate) fn ts_span(ts: &TokenStream) -> Option<Span> {
    let mut iter = ts.trees();
    let first = iter.next()?.span();
    let last = iter.last().map(|x| x.span()).unwrap_or(first);

    Some(first.to(last))
}

fn ts_and_span(ts: &TokenStream, src: impl quote::ToTokens, offset: isize) -> (TokenStream, Span) {
    let ts = corresponding_ts(ts, src, offset);
    let span = ts_span(&ts).expect("unable to get span of empty token stream");
    (ts, span)
}

pub(crate) trait HasTokens {
    fn tokens(&self) -> &TokenStream;
}

pub(crate) trait HasSpan {
    fn span(&self) -> Span;
}

macro_rules! impl_span_tokens {
    ($struct:ty) => {
        impl HasTokens for $struct {
            fn tokens(&self) -> &TokenStream {
                &self.tokens
            }
        }

        impl HasSpan for $struct {
            fn span(&self) -> Span {
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
    fn span(&self) -> Span {
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
    span: Span,
}

impl_span_tokens!(NodeComment<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeDoctype<'a> {
    pub(crate) doctype: &'a node::NodeDoctype,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(NodeDoctype<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeFragment<'a> {
    pub(crate) open_fragment: FragmentOpen<'a>,
    pub(crate) close_fragment: Option<FragmentClose<'a>>,
    pub(crate) children: Option<Children<'a>>,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(NodeFragment<'_>);

#[derive(Debug, Clone)]
pub(crate) struct Children<'a> {
    /// Never empty
    pub(crate) inner: Vec<Node<'a>>,
    tokens: TokenStream,
    span: Span,
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
                Node::from_node(context, &node, ts, offset)
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
    span: Span,
}

impl_span_tokens!(FragmentOpen<'_>);

#[derive(Debug, Clone)]
pub(crate) struct FragmentClose<'a> {
    pub(crate) close_fragment: &'a atoms::FragmentClose,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(FragmentClose<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeElement<'a> {
    pub(crate) element: &'a node::NodeElement,
    pub(crate) open_tag: OpenTag<'a>,
    pub(crate) close_tag: Option<CloseTag<'a>>,
    pub(crate) attributes: Vec<NodeAttributeWithTokens<'a>>,
    pub(crate) children: Option<Children<'a>>,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(NodeElement<'_>);

#[derive(Debug, Clone)]
pub(crate) struct OpenTag<'a> {
    pub(crate) open_tag: &'a atoms::OpenTag,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(OpenTag<'_>);

#[derive(Debug, Clone)]
pub(crate) struct CloseTag<'a> {
    pub(crate) close_tag: &'a atoms::CloseTag,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(CloseTag<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeBlock<'a> {
    pub(crate) block: &'a node::NodeBlock,
    pub(crate) ast: rustc_ast::ptr::P<rustc_ast::Expr>,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(NodeBlock<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeText<'a> {
    pub(crate) text: &'a node::NodeText,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(NodeText<'_>);

#[derive(Debug, Clone)]
pub(crate) struct NodeRawText<'a> {
    pub(crate) raw_text: &'a node::RawText,
    tokens: TokenStream,
    span: Span,
}

impl_span_tokens!(NodeRawText<'_>);

#[derive(Debug, Clone)]
pub(crate) enum NodeAttributeWithTokens<'a> {
    Block(NodeBlock<'a>),
    Attribute(KeyedAttributeWithTokens<'a>),
}

impl NodeAttributeWithTokens<'_> {
    pub(crate) fn from_node<'a>(
        context: &RewriteContext<'_>,
        attr: &'a node::NodeAttribute,
        ts: &TokenStream,
        offset: isize,
    ) -> NodeAttributeWithTokens<'a> {
        match attr {
            node::NodeAttribute::Block(x) => {
                let (tokens, span) = ts_and_span(ts, x, offset);

                let mut parser =
                    rustc_parse::stream_to_parser(context.parse_sess.inner(), tokens.clone(), None);

                NodeAttributeWithTokens::Block(NodeBlock {
                    block: x,
                    ast: parser.parse_expr().unwrap(),
                    tokens,
                    span,
                })
            }
            node::NodeAttribute::Attribute(x) => {
                let (tokens, span) = ts_and_span(ts, x, offset);
                NodeAttributeWithTokens::Attribute(KeyedAttributeWithTokens {
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
pub(crate) struct KeyedAttributeWithTokens<'a> {
    pub(crate) keyed_attribute: &'a node::KeyedAttribute,
    pub(crate) value: Option<Expr<'a>>,
    tokens: TokenStream,
    span: Span,
}

#[derive(Debug, Clone)]
pub(crate) struct Expr<'a> {
    pub(crate) expr: &'a syn::Expr,
    tokens: TokenStream,
    span: Span,
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
                    .map(|attr| NodeAttributeWithTokens::from_node(context, attr, ts, offset))
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

// impl FromInternal<token::Delimiter> for Delimiter {
//     fn from_internal(delim: token::Delimiter) -> Delimiter {
//         match delim {
//             token::Delimiter::Parenthesis => Delimiter::Parenthesis,
//             token::Delimiter::Brace => Delimiter::Brace,
//             token::Delimiter::Bracket => Delimiter::Bracket,
//             token::Delimiter::Invisible => Delimiter::None,
//         }
//     }
// }

// impl FromInternal<token::LitKind> for LitKind {
//     fn from_internal(kind: token::LitKind) -> Self {
//         match kind {
//             token::Byte => LitKind::Byte,
//             token::Char => LitKind::Char,
//             token::Integer => LitKind::Integer,
//             token::Float => LitKind::Float,
//             token::Str => LitKind::Str,
//             token::StrRaw(n) => LitKind::StrRaw(n),
//             token::ByteStr => LitKind::ByteStr,
//             token::ByteStrRaw(n) => LitKind::ByteStrRaw(n),
//             token::CStr => LitKind::CStr,
//             token::CStrRaw(n) => LitKind::CStrRaw(n),
//             token::Err => LitKind::Err,
//             token::Bool => unreachable!(),
//         }
//     }
// }
//
// impl ToInternal<token::LitKind> for LitKind {
//     fn to_internal(self) -> token::LitKind {
//         match self {
//             LitKind::Byte => token::Byte,
//             LitKind::Char => token::Char,
//             LitKind::Integer => token::Integer,
//             LitKind::Float => token::Float,
//             LitKind::Str => token::Str,
//             LitKind::StrRaw(n) => token::StrRaw(n),
//             LitKind::ByteStr => token::ByteStr,
//             LitKind::ByteStrRaw(n) => token::ByteStrRaw(n),
//             LitKind::CStr => token::CStr,
//             LitKind::CStrRaw(n) => token::CStrRaw(n),
//             LitKind::Err => token::Err,
//         }
//     }
// }
//
// impl FromInternal<TokenStream> for Vec<TokenTree<TokenStream, Span, Symbol>> {
//     fn from_internal(stream: TokenStream) -> Self {
//         use rustc_ast::token::*;
//
//         // Estimate the capacity as `stream.len()` rounded up to the next power
//         // of two to limit the number of required reallocations.
//         let mut trees = Vec::with_capacity(stream.len().next_power_of_two());
//         let mut cursor = stream.trees();
//
//         while let Some(tree) = cursor.next() {
//             let (Token { kind, span }, joint) = match tree.clone() {
//                 tokenstream::TokenTree::Delimited(span, delim, tts) => {
//                     let delimiter = pm::Delimiter::from_internal(delim);
//                     trees.push(TokenTree::Group(Group {
//                         delimiter,
//                         stream: Some(tts),
//                         span: DelimSpan {
//                             open: span.open,
//                             close: span.close,
//                             entire: span.entire(),
//                         },
//                     }));
//                     continue;
//                 }
//                 tokenstream::TokenTree::Token(token, spacing) => (token, spacing == Spacing::Joint),
//             };
//
//             // Split the operator into one or more `Punct`s, one per character.
//             // The final one inherits the jointness of the original token. Any
//             // before that get `joint = true`.
//             let mut op = |s: &str| {
//                 assert!(s.is_ascii());
//                 trees.extend(s.bytes().enumerate().map(|(i, ch)| {
//                     let is_final = i == s.len() - 1;
//                     // Split the token span into single chars. Unless the span
//                     // is an unusual one, e.g. due to proc macro expansion. We
//                     // determine this by assuming any span with a length that
//                     // matches the operator length is a normal one, and any
//                     // span with a different length is an unusual one.
//                     let span = if (span.hi() - span.lo()).to_usize() == s.len() {
//                         let lo = span.lo() + BytePos::from_usize(i);
//                         let hi = lo + BytePos::from_usize(1);
//                         span.with_lo(lo).with_hi(hi)
//                     } else {
//                         span
//                     };
//                     TokenTree::Punct(Punct { ch, joint: if is_final { joint } else { true }, span })
//                 }));
//             };
//
//             match kind {
//                 Eq => op("="),
//                 Lt => op("<"),
//                 Le => op("<="),
//                 EqEq => op("=="),
//                 Ne => op("!="),
//                 Ge => op(">="),
//                 Gt => op(">"),
//                 AndAnd => op("&&"),
//                 OrOr => op("||"),
//                 Not => op("!"),
//                 Tilde => op("~"),
//                 BinOp(Plus) => op("+"),
//                 BinOp(Minus) => op("-"),
//                 BinOp(Star) => op("*"),
//                 BinOp(Slash) => op("/"),
//                 BinOp(Percent) => op("%"),
//                 BinOp(Caret) => op("^"),
//                 BinOp(And) => op("&"),
//                 BinOp(Or) => op("|"),
//                 BinOp(Shl) => op("<<"),
//                 BinOp(Shr) => op(">>"),
//                 BinOpEq(Plus) => op("+="),
//                 BinOpEq(Minus) => op("-="),
//                 BinOpEq(Star) => op("*="),
//                 BinOpEq(Slash) => op("/="),
//                 BinOpEq(Percent) => op("%="),
//                 BinOpEq(Caret) => op("^="),
//                 BinOpEq(And) => op("&="),
//                 BinOpEq(Or) => op("|="),
//                 BinOpEq(Shl) => op("<<="),
//                 BinOpEq(Shr) => op(">>="),
//                 At => op("@"),
//                 Dot => op("."),
//                 DotDot => op(".."),
//                 DotDotDot => op("..."),
//                 DotDotEq => op("..="),
//                 Comma => op(","),
//                 Semi => op(";"),
//                 Colon => op(":"),
//                 ModSep => op("::"),
//                 RArrow => op("->"),
//                 LArrow => op("<-"),
//                 FatArrow => op("=>"),
//                 Pound => op("#"),
//                 Dollar => op("$"),
//                 Question => op("?"),
//                 SingleQuote => op("'"),
//
//                 Ident(sym, is_raw) => trees.push(TokenTree::Ident(Ident { sym, is_raw, span })),
//                 Lifetime(name) => {
//                     let ident = symbol::Ident::new(name, span).without_first_quote();
//                     trees.extend([
//                         TokenTree::Punct(Punct { ch: b'\'', joint: true, span }),
//                         TokenTree::Ident(Ident { sym: ident.name, is_raw: false, span }),
//                     ]);
//                 }
//                 Literal(token::Lit { kind, symbol, suffix }) => {
//                     trees.push(TokenTree::Literal(self::Literal {
//                         kind: FromInternal::from_internal(kind),
//                         symbol,
//                         suffix,
//                         span,
//                     }));
//                 }
//                 DocComment(_, attr_style, data) => {
//                     let mut escaped = String::new();
//                     for ch in data.as_str().chars() {
//                         escaped.extend(ch.escape_debug());
//                     }
//                     let stream = [
//                         Ident(sym::doc, false),
//                         Eq,
//                         TokenKind::lit(token::Str, Symbol::intern(&escaped), None),
//                     ]
//                     .into_iter()
//                     .map(|kind| tokenstream::TokenTree::token_alone(kind, span))
//                     .collect();
//                     trees.push(TokenTree::Punct(Punct { ch: b'#', joint: false, span }));
//                     if attr_style == ast::AttrStyle::Inner {
//                         trees.push(TokenTree::Punct(Punct { ch: b'!', joint: false, span }));
//                     }
//                     trees.push(TokenTree::Group(Group {
//                         delimiter: pm::Delimiter::Bracket,
//                         stream: Some(stream),
//                         span: DelimSpan::from_single(span),
//                     }));
//                 }
//
//                 Interpolated(nt) if let NtIdent(ident, is_raw) = *nt => {
//                     trees.push(TokenTree::Ident(Ident { sym: ident.name, is_raw, span: ident.span }))
//                 }
//
//                 Interpolated(nt) => {
//                     let stream = TokenStream::from_nonterminal_ast(&nt);
//                         trees.push(TokenTree::Group(Group {
//                             delimiter: pm::Delimiter::None,
//                             stream: Some(stream),
//                             span: DelimSpan::from_single(span),
//                         }))
//                 }
//
//                 OpenDelim(..) | CloseDelim(..) => unreachable!(),
//                 Eof => unreachable!(),
//             }
//         }
//         trees
//     }
// }
//
// impl ToInternal<tokenstream::TokenTree> for pm::TokenTree {
//     fn to_internal(self) -> tokenstream::TokenTree {
//         match self {
//             pm::TokenTree::Group(x) => tokenstream::TokenTree::Delimited(
//                 tokenstream::DelimSpan{
//                     open: x.span_open().to_internal(),
//                     close: x.span_close().to_internal()
//                 },
//                 x.delimiter().to_internal(),
//                 x.stream().to_internal()
//             ),
//             pm::TokenTree::Ident(x) => tokenstream::TokenTree::token_alone(
//                 token::TokenKind::Ident(Symbol::intern(&x.to_string()), false),
//                 x.span().to_internal(),
//             ),
//             pm::TokenTree::Punct(x) => {
//                 let kind = match x.as_char() {
//                     '=' => token::Eq,
//                     '<' => token::Lt,
//                     '>' => token::Gt,
//                     '!' => token::Not,
//                     '~' => token::Tilde,
//                     '+' => token::BinOp(token::BinOpToken::Plus),
//                     '-' => token::BinOp(token::BinOpToken::Minus),
//                     '*' => token::BinOp(token::BinOpToken::Star),
//                     '/' => token::BinOp(token::BinOpToken::Slash),
//                     '%' => token::BinOp(token::BinOpToken::Percent),
//                     '^' => token::BinOp(token::BinOpToken::Caret),
//                     '&' => token::BinOp(token::BinOpToken::And),
//                     '|' => token::BinOp(token::BinOpToken::Or),
//                     '@' => token::At,
//                     '.' => token::Dot,
//                     ',' => token::Comma,
//                     ';' => token::Semi,
//                     ':' => token::Colon,
//                     '#' => token::Pound,
//                     '$' => token::Dollar,
//                     '?' => token::Question,
//                     '\'' => token::SingleQuote,
//                     _ => unreachable!(),
//                 };
//                 if let pm::Spacing::Joint = x.spacing() {
//                     tokenstream::TokenTree::token_joint(kind, x.span().to_internal())
//                 } else {
//                     tokenstream::TokenTree::token_alone(kind, x.span().to_internal())
//                 }
//             }
//             pm::TokenTree::Literal(x) => {
//                 tokenstream::TokenTree::token_alone(token::TokenKind::Literal(
//                     token::Lit::new(
//                         kind,
//                         symbol,
//                         suffix
//                     )
//                 ), x.0.kind)
//             }
//             pm::TokenTree::Literal(self::Literal {
//                 kind: self::LitKind::Integer,
//                 symbol,
//                 suffix,
//                 span,
//             }) if symbol.as_str().starts_with('-') => {
//                 let minus = BinOp(BinOpToken::Minus);
//                 let symbol = Symbol::intern(&symbol.as_str()[1..]);
//                 let integer = TokenKind::lit(token::Integer, symbol, suffix);
//                 let a = tokenstream::TokenTree::token_alone(minus, span);
//                 let b = tokenstream::TokenTree::token_alone(integer, span);
//                 vec![a, b]
//             }
//             pm::TokenTree::Literal(self::Literal {
//                 kind: self::LitKind::Float,
//                 symbol,
//                 suffix,
//                 span,
//             }) if symbol.as_str().starts_with('-') => {
//                 let minus = BinOp(BinOpToken::Minus);
//                 let symbol = Symbol::intern(&symbol.as_str()[1..]);
//                 let float = TokenKind::lit(token::Float, symbol, suffix);
//                 let a = tokenstream::TokenTree::token_alone(minus, span);
//                 let b = tokenstream::TokenTree::token_alone(float, span);
//                 vec![a, b]
//             }
//             pm::TokenTree::Literal(self::Literal { kind, symbol, suffix, span }) => {
//                 vec![tokenstream::TokenTree::token_alone(
//                     TokenKind::lit(kind.to_internal(), symbol, suffix),
//                     span,
//                 )]
//             }
//         }
//     }
// }
//
// impl ToInternal<TokenStream> for pm::TokenStream {
//     fn to_internal(self) -> TokenStream {
//         let mut ts = TokenStream::new(Vec::new());
//         for tt in self {
//             ts.push_tree(tt.to_internal());
//         }
//
//         ts
//     }
// }
//
// impl ToInternal<Vec<tokenstream::TokenTree>>
//     for TokenTree<TokenStream, Span, Symbol>
// {
//     fn to_internal(self) -> Vec<tokenstream::TokenTree> {
//         use rustc_ast::token::*;
//
//         let tree = self;
//         match tree {
//             TokenTree::Punct(Punct { ch, joint, span }) => {
//                 let kind = match ch {
//                     b'=' => Eq,
//                     b'<' => Lt,
//                     b'>' => Gt,
//                     b'!' => Not,
//                     b'~' => Tilde,
//                     b'+' => BinOp(Plus),
//                     b'-' => BinOp(Minus),
//                     b'*' => BinOp(Star),
//                     b'/' => BinOp(Slash),
//                     b'%' => BinOp(Percent),
//                     b'^' => BinOp(Caret),
//                     b'&' => BinOp(And),
//                     b'|' => BinOp(Or),
//                     b'@' => At,
//                     b'.' => Dot,
//                     b',' => Comma,
//                     b';' => Semi,
//                     b':' => Colon,
//                     b'#' => Pound,
//                     b'$' => Dollar,
//                     b'?' => Question,
//                     b'\'' => SingleQuote,
//                     _ => unreachable!(),
//                 };
//                 vec![if joint {
//                     tokenstream::TokenTree::token_joint(kind, span)
//                 } else {
//                     tokenstream::TokenTree::token_alone(kind, span)
//                 }]
//             }
//             TokenTree::Group(Group { delimiter, stream, span: DelimSpan { open, close, .. } }) => {
//                 vec![tokenstream::TokenTree::Delimited(
//                     tokenstream::DelimSpan { open, close },
//                     delimiter.to_internal(),
//                     stream.unwrap_or_default(),
//                 )]
//             }
//             TokenTree::Ident(self::Ident { sym, is_raw, span }) => {
//                 vec![tokenstream::TokenTree::token_alone(Ident(sym, is_raw), span)]
//             }
//             TokenTree::Literal(self::Literal {
//                 kind: self::LitKind::Integer,
//                 symbol,
//                 suffix,
//                 span,
//             }) if symbol.as_str().starts_with('-') => {
//                 let minus = BinOp(BinOpToken::Minus);
//                 let symbol = Symbol::intern(&symbol.as_str()[1..]);
//                 let integer = TokenKind::lit(token::Integer, symbol, suffix);
//                 let a = tokenstream::TokenTree::token_alone(minus, span);
//                 let b = tokenstream::TokenTree::token_alone(integer, span);
//                 vec![a, b]
//             }
//             TokenTree::Literal(self::Literal {
//                 kind: self::LitKind::Float,
//                 symbol,
//                 suffix,
//                 span,
//             }) if symbol.as_str().starts_with('-') => {
//                 let minus = BinOp(BinOpToken::Minus);
//                 let symbol = Symbol::intern(&symbol.as_str()[1..]);
//                 let float = TokenKind::lit(token::Float, symbol, suffix);
//                 let a = tokenstream::TokenTree::token_alone(minus, span);
//                 let b = tokenstream::TokenTree::token_alone(float, span);
//                 vec![a, b]
//             }
//             TokenTree::Literal(self::Literal { kind, symbol, suffix, span }) => {
//                 vec![tokenstream::TokenTree::token_alone(
//                     TokenKind::lit(kind.to_internal(), symbol, suffix),
//                     span,
//                 )]
//             }
//         }
//     }
// }
//
// // impl ToInternal<rustc_errors::Level> for Level {
// //     fn to_internal(self) -> rustc_errors::Level {
// //         match self {
// //             Level::Error => rustc_errors::Level::Error { lint: false },
// //             Level::Warning => rustc_errors::Level::Warning(None),
// //             Level::Note => rustc_errors::Level::Note,
// //             Level::Help => rustc_errors::Level::Help,
// //             _ => unreachable!("unknown proc_macro::Level variant: {:?}", self),
// //         }
// //     }
// // }

// trait ToProcMacro<T> {
//     fn to_proc_macro(self) -> T;
// }
//
// trait ToRustc<T> {
//     fn to_rustc(self) -> T;
// }
//
// impl ToProcMacro<pm::TokenStream> for TokenStream {
//     fn to_proc_macro(self) -> pm::TokenStream {
//         let mut ts = pm::TokenStream::new();
//         for tt in self.into_trees() {
//             ts.extend(tt.to_proc_macro());
//         }
//
//         ts
//     }
// }
//
// impl ToRustc<TokenStream> for pm::TokenStream {
//     fn to_rustc(self) -> TokenStream {
//         let mut ts = TokenStream::new(Vec::new());
//         for tt in self {
//             ts.push_tree(tt.to_rustc());
//         }
//
//         ts
//     }
// }
//
// impl ToProcMacro<Vec<pm::TokenTree>> for TokenTree {
//     fn to_proc_macro(self: TokenTree) -> Vec<pm::TokenTree> {
//         match self {
//             TokenTree::Token(token, spacing) => {
//                 let span = token.span.to_proc_macro();
//
//                 macro_rules! punct {
//                     ($char:literal) => {{
//                         let mut punct = proc_macro2::TokenTree::Punct(proc_macro2::Punct::new(
//                             $char,
//                             if let rustc_ast::tokenstream::Spacing::Joint = spacing {
//                                 proc_macro2::Spacing::Joint
//                             } else {
//                                 proc_macro2::Spacing::Alone
//                             },
//                         ));
//                         punct.set_span(span);
//                         vec![punct]
//                     }};
//
//                     ($char_1:literal, $char_2:literal) => {{
//                         let mut punct_1 = proc_macro2::TokenTree::Punct(proc_macro2::Punct::new(
//                             $char_1,
//                             proc_macro2::Spacing::Joint,
//                         ));
//                         punct_1.set_span(span);
//                         let mut punct_2 = proc_macro2::TokenTree::Punct(proc_macro2::Punct::new(
//                             $char_2,
//                             if let rustc_ast::tokenstream::Spacing::Joint = spacing {
//                                 proc_macro2::Spacing::Joint
//                             } else {
//                                 proc_macro2::Spacing::Alone
//                             },
//                         ));
//                         punct_2.set_span(span);
//                         vec![punct_1, punct_2]
//                     }};
//
//                     ($char_1:literal, $char_2:literal, $char_3:literal) => {{
//                         let mut punct_1 = proc_macro2::TokenTree::Punct(proc_macro2::Punct::new(
//                             $char_1,
//                             proc_macro2::Spacing::Joint,
//                         ));
//                         punct_1.set_span(span);
//                         let mut punct_2 = proc_macro2::TokenTree::Punct(proc_macro2::Punct::new(
//                             $char_2,
//                             proc_macro2::Spacing::Joint,
//                         ));
//                         punct_2.set_span(span);
//                         let mut punct_3 = proc_macro2::TokenTree::Punct(proc_macro2::Punct::new(
//                             $char_3,
//                             if let rustc_ast::tokenstream::Spacing::Joint = spacing {
//                                 proc_macro2::Spacing::Joint
//                             } else {
//                                 proc_macro2::Spacing::Alone
//                             },
//                         ));
//                         punct_3.set_span(span);
//                         vec![punct_1, punct_2, punct_3]
//                     }};
//                 }
//
//                 match token.kind {
//                     TokenKind::Eq => punct!('='),
//                     TokenKind::Lt => punct!('<'),
//                     TokenKind::Le => punct!('<', '='),
//                     TokenKind::EqEq => punct!('=', '='),
//                     TokenKind::Ne => punct!('!', '='),
//                     TokenKind::Ge => punct!('>', '='),
//                     TokenKind::Gt => punct!('>'),
//                     TokenKind::AndAnd => punct!('&', '&'),
//                     TokenKind::OrOr => punct!('|', '|'),
//                     TokenKind::Not => punct!('!'),
//                     TokenKind::Tilde => punct!('~'),
//                     TokenKind::BinOp(x) => match x {
//                         BinOpToken::Plus => punct!('+'),
//                         BinOpToken::Minus => punct!('-'),
//                         BinOpToken::Star => punct!('*'),
//                         BinOpToken::Slash => punct!('/'),
//                         BinOpToken::Percent => punct!('%'),
//                         BinOpToken::Caret => punct!('^'),
//                         BinOpToken::And => punct!('&'),
//                         BinOpToken::Or => punct!('|'),
//                         BinOpToken::Shl => punct!('<', '<'),
//                         BinOpToken::Shr => punct!('>', '>'),
//                     },
//                     TokenKind::BinOpEq(x) => match x {
//                         BinOpToken::Plus => punct!('+', '='),
//                         BinOpToken::Minus => punct!('-', '='),
//                         BinOpToken::Star => punct!('*', '='),
//                         BinOpToken::Slash => punct!('/', '='),
//                         BinOpToken::Percent => punct!('%', '='),
//                         BinOpToken::Caret => punct!('^', '='),
//                         BinOpToken::And => punct!('&', '='),
//                         BinOpToken::Or => punct!('|', '='),
//                         BinOpToken::Shl => punct!('<', '<', '='),
//                         BinOpToken::Shr => punct!('>', '>', '='),
//                     },
//                     TokenKind::At => punct!('@'),
//                     TokenKind::Dot => punct!('.'),
//                     TokenKind::DotDot => punct!('.', '.'),
//                     TokenKind::DotDotDot => punct!('.', '.', '.'),
//                     TokenKind::DotDotEq => punct!('.', '.', '='),
//                     TokenKind::Comma => punct!(','),
//                     TokenKind::Semi => punct!(';'),
//                     TokenKind::Colon => punct!(':'),
//                     TokenKind::ModSep => punct!(':', ':'),
//                     TokenKind::RArrow => punct!('-', '>'),
//                     TokenKind::LArrow => punct!('<', '-'),
//                     TokenKind::FatArrow => punct!('=', '>'),
//                     TokenKind::Pound => punct!('#'),
//                     TokenKind::Dollar => punct!('$'),
//                     TokenKind::Question => punct!('?'),
//                     TokenKind::SingleQuote => punct!('\''),
//                     TokenKind::OpenDelim(x) => match x {
//                         Delimiter::Parenthesis => punct!('('),
//                         Delimiter::Brace => punct!('['),
//                         Delimiter::Bracket => punct!('{'),
//                         Delimiter::Invisible => Vec::new(),
//                     },
//                     TokenKind::CloseDelim(x) => match x {
//                         Delimiter::Parenthesis => punct!(')'),
//                         Delimiter::Brace => punct!(']'),
//                         Delimiter::Bracket => punct!('}'),
//                         Delimiter::Invisible => Vec::new(),
//                     },
//                     TokenKind::Literal(x) => match x.kind {
//                         LitKind::Bool => {
//                             vec![pm::TokenTree::Ident(pm::Ident::new(&x.to_string(), span))]
//                         }
//                         _ => {
//                             vec![pm::TokenTree::Literal(
//                                 _Literal::_Fallback {
//                                     repr: x.to_string(),
//                                     span: _SpanFallback::from(span),
//                                 }
//                                 .into(),
//                             )]
//                         }
//                     },
//
//                     TokenKind::Ident(sym, is_raw) => {
//                         vec![if is_raw {
//                             pm::TokenTree::Ident(pm::Ident::new_raw(sym.as_str(), span))
//                         } else {
//                             pm::TokenTree::Ident(pm::Ident::new(sym.as_str(), span))
//                         }]
//                     }
//                     TokenKind::Lifetime(_) => todo!(),
//                     TokenKind::Interpolated(_) => todo!(),
//                     TokenKind::DocComment(_, _, _) => todo!(),
//                     TokenKind::Eof => Vec::new(),
//                 }
//             }
//             TokenTree::Delimited(_, _, _) => todo!(),
//         }
//     }
// }
//
// impl ToRustc<TokenTree> for pm::TokenTree {
//     fn to_rustc(self) -> TokenTree {
//         match self {
//             pm::TokenTree::Group(x) => TokenTree::Delimited(
//                 DelimSpan::from_single(x.span().to_rustc()),
//                 x.delimiter().to_rustc(),
//                 x.stream().to_rustc(),
//             ),
//             pm::TokenTree::Ident(x) => {
//                 let span = x.span().to_rustc();
//                 let ident = _Ident::from(x);
//                 TokenTree::token_alone(
//                     TokenKind::Ident(Symbol::intern(ident.sym()), ident.raw()),
//                     span,
//                 )
//             }
//             pm::TokenTree::Punct(x) => {
//                 let kind = match x.as_char() {
//                     '=' => TokenKind::Eq,
//                     '<' => TokenKind::Lt,
//                     // '<=' => TokenKind::Le,
//                     // '==' => TokenKind::EqEq,
//                     // '!=' => TokenKind::Ne,
//                     // '>=' => TokenKind::Ge,
//                     '>' => TokenKind::Gt,
//                     // '&&' => TokenKind::AndAnd,
//                     // '||' => TokenKind::OrOr,
//                     '!' => TokenKind::Not,
//                     '~' => TokenKind::Tilde,
//                     '+' => TokenKind::BinOp(BinOpToken::Plus),
//                     '-' => TokenKind::BinOp(BinOpToken::Minus),
//                     '*' => TokenKind::BinOp(BinOpToken::Star),
//                     '/' => TokenKind::BinOp(BinOpToken::Slash),
//                     '%' => TokenKind::BinOp(BinOpToken::Percent),
//                     '^' => TokenKind::BinOp(BinOpToken::Caret),
//                     '&' => TokenKind::BinOp(BinOpToken::And),
//                     '|' => TokenKind::BinOp(BinOpToken::Or),
//                     // '<<' => TokenKind::BinOp(BinOpToken::Shl),
//                     // '>>' => TokenKind::BinOp(BinOpToken::Shr),
//                     // '+=' => TokenKind::BinOp(BinOpToken::Plus),
//                     // '-=' => TokenKind::BinOp(BinOpToken::Minus),
//                     // '*=' => TokenKind::BinOp(BinOpToken::Star),
//                     // '/=' => TokenKind::BinOp(BinOpToken::Slash),
//                     // '%=' => TokenKind::BinOp(BinOpToken::Percent),
//                     // '^=' => TokenKind::BinOp(BinOpToken::Caret),
//                     // '&=' => TokenKind::BinOp(BinOpToken::And),
//                     // '|=' => TokenKind::BinOp(BinOpToken::Or),
//                     // '<<=' => TokenKind::BinOp(BinOpToken::Shl),
//                     // '>>=' => TokenKind::BinOp(BinOpToken::Shr),
//                     '@' => TokenKind::At,
//                     '.' => TokenKind::Dot,
//                     // '..' => TokenKind::DotDot,
//                     // '...' => TokenKind::DotDotDot,
//                     // '..=' => TokenKind::DotDotEq,
//                     ',' => TokenKind::Comma,
//                     ';' => TokenKind::Semi,
//                     ':' => TokenKind::Colon,
//                     // '::' => TokenKind::ModSep,
//                     // '->' => TokenKind::RArrow,
//                     // '<-' => TokenKind::LArrow,
//                     // '=>' => TokenKind::FatArrow,
//                     '#' => TokenKind::Pound,
//                     '$' => TokenKind::Dollar,
//                     '?' => TokenKind::Question,
//                     '\'' => TokenKind::SingleQuote,
//                     '(' => TokenKind::OpenDelim(Delimiter::Parenthesis),
//                     '[' => TokenKind::OpenDelim(Delimiter::Brace),
//                     '{' => TokenKind::OpenDelim(Delimiter::Bracket),
//                     ')' => TokenKind::CloseDelim(Delimiter::Parenthesis),
//                     ']' => TokenKind::CloseDelim(Delimiter::Brace),
//                     '}' => TokenKind::CloseDelim(Delimiter::Bracket),
//                     _ => unreachable!(),
//                 };
//                 if let pm::Spacing::Joint = x.spacing() {
//                     TokenTree::token_joint(kind, x.span().to_rustc())
//                 } else {
//                     TokenTree::token_alone(kind, x.span().to_rustc())
//                 }
//             }
//             pm::TokenTree::Literal(x) => {
//                 let span = x.span().to_rustc();
//                 TokenTree::token_alone(TokenKind::Literal(x.to_rustc()), span)
//             }
//         }
//     }
// }
//
// impl ToRustc<Lit> for pm::Literal {
//     fn to_rustc(self) -> Lit {
//         let lit: syn::Lit = syn::parse_str(&self.to_string()).unwrap();
//
//         macro_rules! literal {
//             ($kind:ident, $string:expr, $suffix:expr) => {
//                 rustc_ast::token::Lit::new(
//                     rustc_ast::token::LitKind::$kind,
//                     rustc_span::symbol::Symbol::intern(&$string),
//                     Some(rustc_span::symbol::Symbol::intern(&$suffix)),
//                 )
//             };
//             ($kind:ident, $string:expr) => {
//                 rustc_ast::token::Lit::new(
//                     rustc_ast::token::LitKind::$kind,
//                     rustc_span::symbol::Symbol::intern(&$string),
//                     None,
//                 )
//             };
//         }
//
//         match lit {
//             syn::Lit::Str(x) => literal!(Str, x.value(), x.suffix()),
//             syn::Lit::ByteStr(x) => literal!(ByteStr, x.token().to_string(), x.suffix()),
//             syn::Lit::Byte(x) => literal!(Byte, x.token().to_string(), x.suffix()),
//             syn::Lit::Char(x) => literal!(Char, x.token().to_string(), x.suffix()),
//             syn::Lit::Int(x) => literal!(Integer, x.to_string(), x.suffix()),
//             syn::Lit::Float(x) => literal!(Float, x.to_string(), x.suffix()),
//             syn::Lit::Bool(x) => literal!(Bool, x.token().to_string()),
//             syn::Lit::Verbatim(x) => literal!(Err, x.to_string()),
//             _ => unreachable!(),
//         }
//     }
// }
//
// impl ToRustc<Delimiter> for pm::Delimiter {
//     fn to_rustc(self) -> Delimiter {
//         match self {
//             pm::Delimiter::Parenthesis => Delimiter::Parenthesis,
//             pm::Delimiter::Brace => Delimiter::Brace,
//             pm::Delimiter::Bracket => Delimiter::Bracket,
//             pm::Delimiter::None => Delimiter::Invisible,
//         }
//     }
// }
//
// impl ToProcMacro<pm::Span> for Span {
//     fn to_proc_macro(self) -> pm::Span {
//         let span = _Span::Fallback(_SpanFallback {
//             lo: self.lo().to_u32(),
//             hi: self.hi().to_u32(),
//         });
//
//         unsafe { mem::transmute(span) }
//     }
// }
//
// impl ToRustc<Span> for pm::Span {
//     fn to_rustc(self) -> Span {
//         // let file = cx
//         //     .parse_sess
//         //     .inner()
//         //     .source_map()
//         //     .files()
//         //     .first()
//         //     .unwrap()
//         //     .clone();
//         //
//         // let start = file.line_bounds(span.start().line - 1).start.to_usize()
//         //     + file
//         //         .get_line(span.start().line - 1)
//         //         .unwrap()
//         //         .char_indices()
//         //         .nth(span.start().column)
//         //         .unwrap()
//         //         .0;
//         //
//         // let end = file.line_bounds(span.end().line - 1).end.to_usize()
//         //     + file
//         //         .get_line(span.end().line - 1)
//         //         .unwrap()
//         //         .char_indices()
//         //         .nth(span.end().column)
//         //         .unwrap()
//         //         .0;
//
//         // SAFETY: UB but hope it works ¯\_(ツ)_/¯
//         match unsafe { mem::transmute(self) } {
//             _Span::Fallback(x) => Span::new(
//                 BytePos::from_u32(x.lo),
//                 BytePos::from_u32(x.hi),
//                 SyntaxContext::root(),
//                 None,
//             ),
//             _Span::_Compiler(_) => unreachable!(),
//         }
//     }
// }
//
// #[derive(Debug, Clone)]
// pub(crate) enum _Literal {
//     _Compiler(proc_macro::Literal),
//     _Fallback { repr: String, span: _SpanFallback },
// }
//
// impl _Literal {
//     fn repr(&self) -> &String {
//         match self {
//             Self::_Fallback { repr, .. } => repr,
//             Self::_Compiler(_) => unreachable!(),
//         }
//     }
// }
//
// impl From<_Literal> for pm::Literal {
//     fn from(val: _Literal) -> Self {
//         unsafe { mem::transmute(val) }
//     }
// }
//
// #[derive(Debug, Clone)]
// pub(crate) enum _Ident {
//     _Compiler(proc_macro::Ident),
//     _Fallback {
//         sym: String,
//         span: _SpanFallback,
//         raw: bool,
//     },
// }
//
// impl _Ident {
//     fn sym(&self) -> &String {
//         match self {
//             Self::_Fallback { sym, .. } => sym,
//             Self::_Compiler(_) => unreachable!(),
//         }
//     }
//
//     fn raw(&self) -> bool {
//         match self {
//             Self::_Fallback { raw, .. } => *raw,
//             Self::_Compiler(_) => unreachable!(),
//         }
//     }
// }
//
// impl From<pm::Ident> for _Ident {
//     fn from(value: pm::Ident) -> Self {
//         unsafe { mem::transmute(value) }
//     }
// }

#[derive(Debug, Copy, Clone)]
pub(crate) enum _Span {
    _Compiler(proc_macro::Span),
    _Fallback(_SpanFallback),
}

impl _Span {
    pub(crate) fn lo(&self) -> u32 {
        match self {
            Self::_Fallback(x) => x.lo,
            Self::_Compiler(_) => unreachable!(),
        }
    }

    pub(crate) fn hi(&self) -> u32 {
        match self {
            Self::_Fallback(x) => x.hi,
            Self::_Compiler(_) => unreachable!(),
        }
    }
}

impl From<pm::Span> for _Span {
    fn from(value: pm::Span) -> Self {
        // SAFETY: UB but it works ¯\_(ツ)_/¯
        unsafe { mem::transmute(value) }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct _SpanFallback {
    pub(crate) lo: u32,
    pub(crate) hi: u32,
}

impl From<pm::Span> for _SpanFallback {
    fn from(value: pm::Span) -> Self {
        let span = _Span::from(value);

        match span {
            _Span::_Fallback(x) => x,
            _Span::_Compiler(_) => unreachable!(),
        }
    }
}

// #[derive(Clone)]
// pub struct Group<TokenStream, Span> {
//     pub delimiter: Delimiter,
//     pub stream: Option<TokenStream>,
//     pub span: DelimSpan<Span>,
// }
//
// #[derive(Clone)]
// pub struct Punct {
//     pub ch: u8,
//     pub joint: bool,
//     pub span: std::num::NonZeroU32,
// }
//
// #[derive(Copy, Clone, Eq, PartialEq)]
// pub struct Ident<Symbol> {
//     pub sym: Symbol,
//     pub is_raw: bool,
//     pub span: std::num::NonZeroU32,
// }
//
// #[derive(Clone, Eq, PartialEq)]
// pub struct Literal<Symbol> {
//     pub kind: LitKind,
//     pub symbol: Symbol,
//     pub suffix: Option<Symbol>,
//     pub span: std::num::NonZeroU32,
// }
