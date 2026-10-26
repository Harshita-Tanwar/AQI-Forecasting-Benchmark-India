// Layout helpers for an IEEE-conference-style A4 Word manuscript (two columns, Times New Roman).
const fs = require("fs");
const {
  Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, WidthType, BorderStyle, AlignmentType,
  TabStopType, HighlightColor, VerticalAlign, Math: OMath, MathRun, MathFraction, MathRadical,
  MathSum, MathSubScript, MathSuperScript,
} = require("docx");

const FONT = "Times New Roman";
const PAGE = { width: 11906, height: 16838 };
const MARGIN = { top: 1080, bottom: 1440, left: 1022, right: 1022 };
const COL_SPACE = 346;
const FULL_W = PAGE.width - MARGIN.left - MARGIN.right;   // 9862
const COL_W = Math.floor((FULL_W - COL_SPACE) / 2);       // 4758
const pt = (n) => n * 2;                                    // docx sizes are half-points

// Inline markup: **bold**, *italic*, ^sup^, ~sub~, [[placeholder]] (yellow highlight)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^[^^]+\^|~[^~]+~|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ font: FONT, ...base, text: t.slice(2, -2), bold: true }));
    else if (t.startsWith("[[")) out.push(new TextRun({ font: FONT, ...base, text: `[${t.slice(2, -2)}]`, highlight: HighlightColor.YELLOW }));
    else if (t.startsWith("*")) out.push(new TextRun({ font: FONT, ...base, text: t.slice(1, -1), italics: true }));
    else if (t.startsWith("^")) out.push(new TextRun({ font: FONT, ...base, text: t.slice(1, -1), superScript: true }));
    else out.push(new TextRun({ font: FONT, ...base, text: t.slice(1, -1), subScript: true }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 100, line: 240 },
  indent: { firstLine: 202 },
  ...opts,
  children: runs(text, { size: pt(10) }),
});

const H1 = (num, text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 200, after: 100 },
  keepNext: true,
  children: [new TextRun({ text: `${num}. `, font: FONT, size: pt(10) }),
             new TextRun({ text, font: FONT, size: pt(10), smallCaps: true })],
});

const H2 = (letter, text) => new Paragraph({
  spacing: { before: 120, after: 60 },
  keepNext: true,
  children: [new TextRun({ text: `${letter}. ${text}`, font: FONT, size: pt(10), italics: true })],
});

function bullets(items) {
  return items.map((t) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 60, line: 240 },
    indent: { left: 288, hanging: 200 },
    children: [new TextRun({ text: "•\t", font: FONT, size: pt(10) }), ...runs(t, { size: pt(10) })],
    tabStops: [{ type: TabStopType.LEFT, position: 288 }],
  }));
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const figNums = {};
const figKey = (file) => require("path").basename(file, ".png");
function figRef(key) {
  if (!(key in figNums)) figNums[key] = Object.keys(figNums).length + 1;
  return `Fig. ${figNums[key]}`;
}
function figure(file, caption, widthIn = 3.3) {
  const figNo = Number(figRef(figKey(file)).slice(5));
  if (!fs.existsSync(file)) {
    return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
      children: runs(`[[Fig. ${figNo} is generated when the experiments finish]] ${caption}`, { size: pt(8) }) })];
  }
  const { w, h } = pngSize(file);
  const px = widthIn * 96;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: Math.round(px), height: Math.round(px * h / w) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
      children: [new TextRun({ text: `Fig. ${figNo}. `, font: FONT, size: pt(8) }), ...runs(caption, { size: pt(8) })],
    }),
  ];
}

const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];
let tabNo = 0;
const rule = (size) => ({ style: BorderStyle.SINGLE, size, color: "000000" });
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

// Booktabs-style table: rules above/below the header and at the bottom only.
function table(caption, header, rows, widths, opts = {}) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const fs8 = pt(opts.fontSize || 8);
  const align = opts.align || header.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER));
  const cell = (text, i, r, isHeader, nRows) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    margins: { top: 20, bottom: 20, left: opts.pad ?? 50, right: opts.pad ?? 50 },
    verticalAlign: VerticalAlign.CENTER,
    borders: {
      top: isHeader ? rule(8) : none,
      bottom: isHeader ? rule(4) : (r === nRows - 1 ? rule(8) : none),
      left: none, right: none,
    },
    children: [new Paragraph({ alignment: align[i], spacing: { after: 0 },
      children: runs(String(text), { size: fs8, bold: isHeader || undefined }) })],
  });
  const tbl = new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: [
      new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, -1, true, rows.length)) }),
      ...rows.map((row, r) => new TableRow({ children: row.map((t, i) => cell(t, i, r, false, rows.length)) })),
    ],
  });
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}. `, font: FONT, size: pt(8) }),
                 new TextRun({ text: caption, font: FONT, size: pt(8), smallCaps: true })],
    }),
    tbl,
    ...(opts.note ? [new Paragraph({ spacing: { before: 40, after: 160 }, children: runs(opts.note, { size: pt(7), italics: false }) })]
                  : [new Paragraph({ spacing: { after: 120 }, children: [] })]),
  ];
}

// Numbered display equation: centred maths, number flush right.
function equation(mathChildren, n) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(COL_W / 2) },
               { type: TabStopType.RIGHT, position: COL_W }],
    children: [new TextRun({ text: "\t", font: FONT }), new OMath({ children: mathChildren }),
               new TextRun({ text: `\t(${n})`, font: FONT, size: pt(10) })],
  });
}

const M = {
  r: (t) => new MathRun(t),
  frac: (num, den) => new MathFraction({ numerator: num, denominator: den }),
  sqrt: (children) => new MathRadical({ children }),
  sum: (children) => new MathSum({ children, subScript: [new MathRun("i=1")], superScript: [new MathRun("n")] }),
  sub: (base, sub) => new MathSubScript({ children: [new MathRun(base)], subScript: [new MathRun(sub)] }),
  sup: (children, sup) => new MathSuperScript({ children, superScript: [new MathRun(sup)] }),
};

function references(list) {
  return list.map((ref, i) => new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 50, line: 220 },
    indent: { left: 400, hanging: 400 },
    tabStops: [{ type: TabStopType.LEFT, position: 400 }],
    children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: pt(8) }), ...runs(ref, { size: pt(8) })],
  }));
}


// ----- pieces that differ between layouts -----
const sec = (roman, letter) => (letter ? `Section ${roman}-${letter}` : `Section ${roman}`);
const tab = (roman) => `Table ${roman}`;

function abstract(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 120 },
    children: [new TextRun({ text: "Abstract: ", bold: true, italics: true, font: FONT, size: pt(9) }),
               ...runs(text, { size: pt(9), bold: true })],
  });
}
function keywords(list) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 },
    children: [new TextRun({ text: "Keywords: ", bold: true, italics: true, font: FONT, size: pt(9) }),
               ...runs(list.join(", "), { size: pt(9), bold: true })],
  });
}
function ack(numeral, text) {
  return [H1(numeral, "Acknowledgment"), P(text, { indent: { firstLine: 0 } })];
}

function authorCell(lines, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders: { top: none, bottom: none, left: none, right: none },
    margins: { top: 40, bottom: 120, left: 40, right: 40 },
    children: lines.map((t, i) => new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: runs(`[[${t}]]`, { size: pt(i === 0 ? 11 : 10) }),
    })),
  });
}

function front(title, authors) {
  const w3 = Math.floor(FULL_W / 3);
  const pad = Math.floor((FULL_W - w3 * 2) / 2);
  const noBorders = { top: none, bottom: none, left: none, right: none, insideHorizontal: none, insideVertical: none };
  const blank = (w) => new TableCell({ width: { size: w, type: WidthType.DXA },
    borders: { top: none, bottom: none, left: none, right: none }, children: [new Paragraph({ children: [] })] });
  const lines = (a) => [a.name, a.dept, a.inst, a.city, a.email];
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
      children: [new TextRun({ text: title, font: FONT, size: pt(24) })] }),
    new Table({ width: { size: w3 * 3, type: WidthType.DXA }, columnWidths: [w3, w3, w3], alignment: AlignmentType.CENTER, borders: noBorders,
      rows: [new TableRow({ children: authors.slice(0, 3).map((a) => authorCell(lines(a), w3)) })] }),
    new Paragraph({ spacing: { after: 0, line: 120 }, children: [] }),   // keeps Word from merging the two tables
    new Table({ width: { size: w3 * 2 + pad * 2, type: WidthType.DXA }, columnWidths: [pad, w3, w3, pad], alignment: AlignmentType.CENTER, borders: noBorders,
      rows: [new TableRow({ children: [blank(pad), ...authors.slice(3, 5).map((a) => authorCell(lines(a), w3)), blank(pad)] })] }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

function makeDocument({ title, authors, body, refs, Document, SectionType }) {
  return new Document({
    creator: "[Authors]", title,
    styles: { default: { document: { run: { font: FONT, size: pt(10) } } } },
    sections: [
      { properties: { page: { size: PAGE, margin: MARGIN } }, children: front(title, authors) },
      { properties: { type: SectionType.CONTINUOUS, page: { size: PAGE, margin: MARGIN },
                      column: { count: 2, space: COL_SPACE, equalWidth: true } },
        children: [...body, H1("IX", "References"), ...references(refs)] },
    ],
  });
}

const include = () => true;
module.exports = { include, style: "ieee", sec, tab, abstract, keywords, ack, front, makeDocument, figRef, FONT, PAGE, MARGIN, COL_SPACE, FULL_W, COL_W, pt, runs, P, H1, H2, bullets, figure, table, equation, M, references };
