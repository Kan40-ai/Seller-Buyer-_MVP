from flask import Flask, request, jsonify, send_from_directory, render_template
from pathlib import Path
import pickle
import pandas as pd  # needed to rebuild DataFrames and handle dates

from matching_engine import SellerBuyerConfig, SellerBuyerMatching
from call_report_etl import CallReport5300ETL


# -----------------------------------------------------------------------------
# Paths and Flask app
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "Templates"),  # your folder name is "Templates"
    static_folder=str(BASE_DIR / "static"),
)

UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
UPLOAD_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Load preloaded 5300 data from pickle files
# -----------------------------------------------------------------------------
with open(BASE_DIR / "call_summary_preloaded.pkl", "rb") as f:
    call_summary_preloaded = pickle.load(f)

with open(BASE_DIR / "call_parts_preloaded.pkl", "rb") as f:
    call_parts_preloaded = pickle.load(f)

# -----------------------------------------------------------------------------
# Rebuild call_summary_preloaded:
# column-wise dict -> DataFrame -> {(inst_id, as_of): row}
# -----------------------------------------------------------------------------
call_summary_df = pd.DataFrame(call_summary_preloaded)
call_summary_df["as_of_date"] = pd.to_datetime(
    call_summary_df["as_of_date"]
).dt.normalize()

summary_dict = {}
for _, row in call_summary_df.iterrows():
    inst_id = row["institution_id"]
    as_of = row["as_of_date"]
    summary_dict[(inst_id, as_of)] = row

call_summary_preloaded = summary_dict
# Result: Dict[(inst_id, as_of_date), summary_row]



# -----------------------------------------------------------------------------
# Rebuild call_parts_preloaded:
# column-wise dict/DataFrame -> add as_of_date=2022-06-30 ->
# groupby(inst_id, as_of) -> {(inst_id, as_of): parts_df}
# -----------------------------------------------------------------------------
if isinstance(call_parts_preloaded, dict):
    parts_df = pd.DataFrame(call_parts_preloaded)
else:
    parts_df = call_parts_preloaded

# For this vintage, all parts are as of 2022-06-30
if "as_of_date" not in parts_df.columns:
    parts_df["as_of_date"] = pd.to_datetime("2022-06-30")

parts_df["as_of_date"] = pd.to_datetime(parts_df["as_of_date"]).dt.normalize()

parts_dict = {}
for (inst_id, as_of), group in parts_df.groupby(["institution_id", "as_of_date"]):
    parts_dict[(inst_id, as_of)] = group.reset_index(drop=True)

call_parts_preloaded = parts_dict
# Result: Dict[(inst_id, as_of_date), parts_df]

ltd_lookup = dict(
    zip(call_summary_df["institution_id"], call_summary_df["ltd"])
)

size_lookup = dict(
    zip(
        call_summary_df["institution_id"],
        zip(call_summary_df["total_assets"], call_summary_df["size_band"])
    )
)

state_lookup = dict(
    zip(call_summary_df["institution_id"], call_summary_df["state"])
)

# -----------------------------------------------------------------------------
# Static configuration lookups (TEMP PLACEHOLDERS – fill with real data later)
# -----------------------------------------------------------------------------
geo_neighbors = {
    "TX": ["OK", "NM", "AR", "LA"],
    "OK": ["TX", "KS", "AR", "MO"],
    # ...
}

state_map = {
    # e.g. "TEXAS": "TX", "OKLAHOMA": "OK", ...
}

# -----------------------------------------------------------------------------
# Create SellerBuyerMatching instance
# -----------------------------------------------------------------------------
config = SellerBuyerConfig(
    call_report_etl_cls=CallReport5300ETL,
    preloaded_summaries=call_summary_preloaded,
    preloaded_parts=call_parts_preloaded,
    geo_neighbors=geo_neighbors,
    state_map=state_map,
)

sbm = SellerBuyerMatching(config=config)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.post("/upload_call_report")
def upload_call_report():
    """
    1. Accept uploaded Excel call report.
    2. Run CallReport5300ETL.
    3. Run full SellerBuyerMatching pipeline.
    4. Generate PDF report.
    5. Return JSON with PDF URL.
    """
    file = request.files.get("call_report")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save uploaded file
    upload_path = UPLOAD_DIR / file.filename
    file.save(upload_path)

    # ---------------- 1. ETL for new CU ----------------
    try:
        new_cu = sbm.run_call_report_etl(upload_path)
    except Exception as e:
        print("Error in run_call_report_etl:", repr(e))
        return jsonify({"error": "Failed to run call report ETL."}), 500

    inst_id = new_cu["institution_id"]
    as_of = new_cu["as_of_date"]

    # new_cu["summary"] should be the same schema as rows in call_summary_df
    ltd_lookup[inst_id] = new_cu["summary"]["ltd"]
    size_lookup[inst_id] = (
        new_cu["summary"]["total_assets"],
        new_cu["summary"]["size_band"],
    )
    state_lookup[inst_id] = new_cu["summary"]["state"]

    # ---------------- 2. LTD classification -------------
    try:
        role, ltd = sbm.classify_ltd(new_cu["summary"])
    except Exception as e:
        print("Error in classify_ltd:", repr(e))
        return jsonify({"error": "Failed to classify LTD / role."}), 500

    # ---------------- 3. Opposite-side pool -------------
    try:
        pool = sbm.build_opposite_pool(
            new_inst_id=inst_id,
            as_of_date=as_of,
            new_role=role,
        )
    except Exception as e:
        print("Error in build_opposite_pool:", repr(e))
        return jsonify({"error": "Failed to build opposite-side pool."}), 500

    if not pool:
        return jsonify({"error": "No opposite-side candidates found for this quarter."}), 400

    # ---------------- 4. Activity scores ----------------
    try:
        scores_long, pair_scores = sbm.compute_activity_scores(
            new_cu=new_cu,
            opposite_pool=pool,
            new_role=role,
        )
    except Exception as e:
        print("Error in compute_activity_scores:", repr(e))
        return jsonify({"error": "Failed to compute activity scores."}), 500

    if scores_long.empty or pair_scores.empty:
        return jsonify({"error": "No activity data available for matching."}), 400

    # ---------------- 5. Activity-based pairs -----------
    try:
        if role == "BUYER":
            buyer_id = inst_id
        else:
            # new CU is SELLER; pick a buyer from pool for MVP
            buyer_id = None
            for cu in pool:
                if cu["side"] == "BUYER":
                    buyer_id = cu["institution_id"]
                    break
            if buyer_id is None:
                return jsonify({"error": "No buyer counterparties found in pool."}), 400

        activity_pairs = sbm.build_activity_pairs(
            buyer_id=buyer_id,
            scores_long=scores_long,
            pair_scores=pair_scores,
        )
    except Exception as e:
        print("Error in build_activity_pairs:", repr(e))
        return jsonify({"error": "Failed to build activity pairs."}), 500

    if activity_pairs.empty:
        return jsonify({"error": "No valid pairs after activity matching."}), 400

    # ---------------- 6. LTD / size / geo refinements ---
    try:
        ltd_pairs = sbm.apply_ltd_band_refinement(
            pairs_df=activity_pairs,
            ltd_lookup=ltd_lookup,
            buyer_label="Caprock",
        )
    except Exception as e:
        print("Error in apply_ltd_band_refinement:", repr(e))
        return jsonify({"error": "Failed in LTD band refinement."}), 500

    if ltd_pairs.empty:
        return jsonify({"error": "No pairs remained after LTD refinement."}), 400

    try:
        size_pairs = sbm.apply_size_band_refinement(
            pairs_df=ltd_pairs,
            size_lookup=size_lookup,
            buyer_label="Caprock",
        )
    except Exception as e:
        print("Error in apply_size_band_refinement:", repr(e))
        return jsonify({"error": "Failed in size band refinement."}), 500

    try:
        geo_pairs = sbm.apply_geo_refinement(
            pairs_df=size_pairs,
            state_lookup=state_lookup,
            buyer_label="Caprock",
        )
    except Exception as e:
        print("Error in apply_geo_refinement:", repr(e))
        return jsonify({"error": "Failed in geo refinement."}), 500

    if geo_pairs.empty:
        return jsonify({"error": "No pairs remained after geo refinement."}), 400

    # ---------------- 7. Final ranking ------------------
    try:
        ranked = sbm.build_ranked_matches(geo_pairs)
    except Exception as e:
        print("Error in build_ranked_matches:", repr(e))
        return jsonify({"error": "Failed to build ranked matches."}), 500

    if ranked.empty:
        return jsonify({"error": "No ranked matches available."}), 400

    # ---------------- 8. PDF report ---------------------
    try:
        pdf_name = "Caprock_SellerBuyer_Matching_Report.pdf"
        pdf_path = REPORT_DIR / pdf_name
        sbm.generate_pdf_report(
            ranked_matches=ranked,
            output_path=str(pdf_path),
            buyer_label="Caprock",
        )
    except Exception as e:
        print("Error in generate_pdf_report:", repr(e))
        return jsonify({"error": "Failed to generate PDF report."}), 500

    return jsonify({"pdf_url": f"/reports/{pdf_name}"})


@app.get("/reports/<name>")
def get_report(name: str):
    return send_from_directory(REPORT_DIR, name)


import os

if __name__ == "__main__":
    # Get port from environment variable, or default to 10000 for Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

