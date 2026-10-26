// Builds the Word manuscript from results/ (via paper_numbers.json) and figures/paper/.
//   node paper/build_paper.js   -> IEEE two-column manuscript (paper/AQI_Forecasting_Paper.docx)
// Run `python -m src.export_paper_numbers` first whenever the results change.
const fs = require("fs");
const path = require("path");
const { Document, Packer, SectionType } = require("docx");

const style = process.argv.includes("--style") ? process.argv[process.argv.indexOf("--style") + 1] : "ieee";
if (style !== "ieee" && !fs.existsSync(path.join(__dirname, `${style}.js`))) {
  console.error(`Layout "${style}" is not available in this repository; only the IEEE layout is included.`);
  process.exit(1);
}
const L = require(`./${style}`);
const { cite, bibliography, unused } = require("./references");
const { content } = require("./content");

const ROOT = path.resolve(__dirname, "..");
const N = JSON.parse(fs.readFileSync(path.join(__dirname, "paper_numbers.json"), "utf8"));
const FIG = (name) => path.join(ROOT, "figures", "paper", `${name}.png`);
const OUT = path.join(__dirname, style === "springer" ? "AQI_Forecasting_Paper_Springer.docx" : "AQI_Forecasting_Paper.docx");

// Placeholders, highlighted in the output; fill in before submission.
const AUTHORS = [1, 2, 3, 4].map((i) => ({ name: `Author ${i} Name`, dept: "Department", inst: "Institution", city: "City, Country", email: "email" }))
  .concat([{ name: "Professor Name (Designation)", dept: "Department", inst: "Institution", city: "City, Country", email: "email" }]);

const body = content.body({ L, N, cite, FIG });
const refs = bibliography(style);
const missing = unused();
if (missing.length) console.warn("Unused references (not printed):", missing.join(", "));

const doc = L.makeDocument({ title: content.title, authors: AUTHORS, body, refs, Document, SectionType });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, `(${refs.length} references)`);
});
