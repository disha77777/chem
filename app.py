import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from excel import export_research, parse_research_response, research_query
from datacleaner import apply_operations, clean_pipeline, interpret_request, summarize
from presentation import powerpoint, presentation
from reader import extract_text_from_image, parse_ocr_to_dataframe, process_image_ocr
from research_assistant import research_verified
from research_planner import generate_research_plan


BASE_DIR = Path(__file__).resolve().parent


def load_key_env():
    """Load local key.env values without requiring an additional package."""
    for filename in ("key.env", ".env"):
        env_file = BASE_DIR / filename
        if not env_file.exists():
            continue

        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_key_env()


def friendly_error_message(error):
    """Translate a Gemini rate-limit error into an actionable message."""
    if "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error):
        return (
            "Gemini rate limit reached. Wait a bit before trying again, or check your "
            "plan and usage at https://ai.dev/rate-limit."
        )
    return str(error)


st.set_page_config(page_title="Multipanel", page_icon="📊", layout="centered")
st.title("Multipanel")
st.write("Create presentations and research workbooks from one local workspace.")

presentation_tab, excel_tab, verified_tab, research_tab, cleaning_tab, reader_tab = st.tabs(["Presentation", "Excel research", "Verified Research", "Research Planning", "Data cleaning", "Reader"])


with presentation_tab:
    with st.form("presentation_form"):
        user_query = st.text_area(
            "Topic or thesis statement",
            placeholder="Remote work improves productivity when companies use outcome-based management.",
            height=120,
        )
        slide_count = st.slider("Number of slides", min_value=2, max_value=15, value=6)
        audience = st.text_input("Audience", value="general audience")
        tone = st.text_input("Tone", value="clear and persuasive")
        submitted = st.form_submit_button("Create presentation", type="primary")

    if submitted:
        if not user_query.strip():
            st.error("Enter a topic or thesis statement first.")
        else:
            try:
                with st.spinner("Creating your presentation plan..."):
                    presentation_info = powerpoint(
                        user_query=user_query.strip(),
                        slide_count=slide_count,
                        audience=audience.strip() or "general audience",
                        tone=tone.strip() or "clear and persuasive",
                    )

                output_path = BASE_DIR / "report.pptx"
                presentation(presentation_info, output_path)

                st.success("Presentation created.")
                st.subheader(presentation_info.get("title", "Presentation"))
                for index, slide_data in enumerate(presentation_info.get("slides", []), start=1):
                    with st.expander(f"Slide {index}: {slide_data.get('title', '')}"):
                        for bullet in slide_data.get("bullets", []):
                            st.markdown(f"- {bullet}")

                st.download_button(
                    "Download PowerPoint",
                    data=output_path.read_bytes(),
                    file_name="report.pptx",
                    mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                )
            except Exception as error:
                st.error(f"Could not create the presentation: {friendly_error_message(error)}")

with excel_tab:
    st.write("Research a question and export the sourced findings as a formatted workbook.")
    with st.form("excel_form"):
        research_request = st.text_area(
            "Research question",
            placeholder="Compare renewable energy investment trends in the US and Europe from 2020 to 2025.",
            height=120,
        )
        excel_submitted = st.form_submit_button("Create Excel workbook", type="primary")

    if excel_submitted:
        if not research_request.strip():
            st.error("Enter a research question first.")
        else:
            try:
                with st.spinner("Researching and building your workbook..."):
                    response_text = research_query(research_request.strip())
                    summary, sources, dataframe = parse_research_response(response_text)
                    output_path = BASE_DIR / "ai_table.xlsx"
                    export_research(
                        dataframe,
                        excel_file=output_path,
                        summary=summary,
                        sources=sources,
                        query=research_request.strip(),
                    )

                st.success("Excel workbook created.")
                st.subheader("Research preview")
                st.dataframe(dataframe, use_container_width=True)
                if summary:
                    st.write(summary)
                st.download_button(
                    "Download Excel workbook",
                    data=output_path.read_bytes(),
                    file_name="ai_table.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as error:
                st.error(f"Could not create the Excel workbook: {friendly_error_message(error)}")

with verified_tab:
    st.write("Research current information with live sources, claim-level citations, and explicit uncertainty.")
    with st.form("verified_research_form"):
        verified_question = st.text_area(
            "What do you need to verify?",
            placeholder="What were the latest global renewable energy capacity additions, according to primary sources?",
            height=120,
        )
        date_range = st.text_input("Date requirement", value="latest available")
        source_preference = st.text_input("Preferred sources", value="credible primary sources")
        verified_submitted = st.form_submit_button("Research and verify", type="primary")

    if verified_submitted:
        if not verified_question.strip():
            st.error("Enter a research question first.")
        else:
            try:
                with st.spinner("Searching current sources and checking claims..."):
                    report = research_verified(
                        verified_question.strip(),
                        date_range=date_range.strip() or "latest available",
                        source_preference=source_preference.strip() or "credible primary sources",
                    )

                st.success(f"Verified report generated as of {report['as_of']}.")
                st.subheader("Answer")
                st.write(report["answer"])
                st.caption(f"Checked at {report['verified_at']}")

                st.subheader("Evidence ledger")
                source_map = {source["id"]: source for source in report["sources"]}
                for claim in report["claims"]:
                    confidence = claim["confidence"].upper()
                    st.markdown(f"**{confidence} confidence · {claim['kind']}**")
                    st.write(claim["text"])
                    cited_sources = [source_map[source_id] for source_id in claim["source_ids"]]
                    for source in cited_sources:
                        st.markdown(
                            f"Source: [{source['title']}]({source['url']}) · "
                            f"{source['publisher']} · published {source['published_date']}"
                        )

                st.subheader("Sources")
                for source in report["sources"]:
                    st.markdown(
                        f"- [{source['title']}]({source['url']}) — {source['publisher']} "
                        f"({source['published_date']})"
                    )

                if report["limitations"]:
                    st.subheader("Limitations")
                    for limitation in report["limitations"]:
                        st.warning(limitation)

                st.download_button(
                    "Download verified report",
                    data=json.dumps(report, indent=2),
                    file_name="verified_research_report.json",
                    mime="application/json",
                )
            except Exception as error:
                st.error(f"Could not verify this research request: {friendly_error_message(error)}")

with research_tab:
    st.write("Create a comprehensive research project plan for your thesis or research project.")
    with st.form("research_form"):
        thesis_statement = st.text_area(
            "Thesis statement or research question",
            placeholder="How does artificial intelligence impact job market dynamics in developing economies?",
            height=120,
        )
        field_of_study = st.text_input("Field of study", value="Research")
        research_type = st.selectbox("Research type", ["Thesis/Dissertation", "Research Paper", "Grant Proposal", "Capstone Project"])
        duration_months = st.slider("Expected duration (months)", min_value=3, max_value=48, value=12)
        research_submitted = st.form_submit_button("Generate research plan", type="primary")

    if research_submitted:
        if not thesis_statement.strip():
            st.error("Enter a thesis statement or research question first.")
        else:
            try:
                with st.spinner("Generating your research plan..."):
                    plan = generate_research_plan(
                        thesis_statement=thesis_statement.strip(),
                        field_of_study=field_of_study.strip() or "Research",
                        research_type=research_type,
                        duration_months=duration_months,
                    )

                st.success("Research plan created!")
                st.subheader(plan.get("title", "Research Project Plan"))

                # Display thesis statement
                st.markdown("**Thesis Statement:**")
                st.markdown(plan.get("thesis_statement", ""))

                # Display objectives
                st.subheader("Research Objectives")
                for obj in plan.get("objectives", []):
                    with st.expander(f"📌 {obj.get('title', '')}"):
                        st.write(obj.get("description", ""))

                # Display literature review
                lit_review = plan.get("literature_review", {})
                st.subheader("Literature Review")
                st.write("**Scope:**", lit_review.get("scope", ""))
                st.write("**Key themes:**", ", ".join(lit_review.get("key_themes", [])))
                st.write("**Sources to explore:**", ", ".join(lit_review.get("sources_to_explore", [])))

                # Display methodology
                methodology = plan.get("methodology", {})
                st.subheader("Methodology")
                st.write("**Research Design:**", methodology.get("research_design", ""))
                st.write("**Data Collection Methods:**")
                for method in methodology.get("data_collection", []):
                    st.write(f"- {method}")
                st.write("**Analysis Approach:**", methodology.get("analysis_approach", ""))
                st.write("**Sample Size/Scope:**", methodology.get("sample_size_or_scope", ""))

                # Display timeline
                st.subheader("Project Timeline")
                timeline_data = []
                for phase in plan.get("timeline", []):
                    timeline_data.append({
                        "Phase": phase.get("phase", ""),
                        "Description": phase.get("description", ""),
                        "Duration (weeks)": phase.get("duration_weeks", 0)
                    })
                if timeline_data:
                    st.dataframe(pd.DataFrame(timeline_data), use_container_width=True)

                # Display resources
                st.subheader("Resources Needed")
                for resource in plan.get("resources_needed", []):
                    st.write(f"**{resource.get('resource', '')}:** {resource.get('justification', '')}")

                # Display expected outcomes
                st.subheader("Expected Outcomes & Deliverables")
                for outcome in plan.get("expected_outcomes", []):
                    st.write(f"✓ {outcome}")

                # Display potential challenges
                if plan.get("potential_challenges"):
                    st.subheader("Potential Challenges & Mitigation")
                    for challenge in plan.get("potential_challenges", []):
                        with st.expander(f"⚠️ {challenge.get('challenge', '')}"):
                            st.write(f"**Mitigation:** {challenge.get('mitigation', '')}")

                # Display starting references
                st.subheader("Key References to Start With")
                for ref in plan.get("references_to_start", []):
                    st.write(f"- {ref}")

                # Download plan as JSON
                st.download_button(
                    "Download plan as JSON",
                    data=json.dumps(plan, indent=2),
                    file_name="research_plan.json",
                    mime="application/json",
                )
            except Exception as error:
                st.error(f"Could not create the research plan: {friendly_error_message(error)}")

with cleaning_tab:
    st.write("Upload a CSV or Excel file, then chat to tell it what to clean or change.")
    uploaded_file = st.file_uploader("Data file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        if st.session_state.get("cleaning_source_name") != uploaded_file.name:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)

                st.session_state["cleaning_source_name"] = uploaded_file.name
                st.session_state["cleaning_df"] = clean_pipeline(raw_df)
                st.session_state["cleaning_chat"] = [
                    {"role": "assistant", "content": f"Loaded and auto-cleaned {len(raw_df)} rows. Ask me to make further changes, e.g. \"drop rows where amount is null\" or \"rename region to area\"."}
                ]
            except Exception as error:
                st.error(f"Could not read this file: {friendly_error_message(error)}")
                st.session_state.pop("cleaning_df", None)

    cleaned_df = st.session_state.get("cleaning_df")
    if cleaned_df is not None:
        st.subheader("Current data preview")
        st.dataframe(cleaned_df.head(50), use_container_width=True)

        with st.expander("Column summary"):
            stats = summarize(cleaned_df)
            st.write({"rows": stats["rows"], "columns": stats["columns"]})
            st.write("Missing values by column:")
            st.json(stats["missing_by_column"])
            st.write("Data types:")
            st.json(stats["dtypes"])

        st.download_button(
            "Download cleaned CSV",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

        st.subheader("Chat with your data")
        for message in st.session_state.get("cleaning_chat", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        chat_request = st.chat_input("What would you like to change?")
        if chat_request:
            st.session_state["cleaning_chat"].append({"role": "user", "content": chat_request})
            try:
                with st.spinner("Applying your request..."):
                    operations = interpret_request(cleaned_df, chat_request)
                    if not operations:
                        reply = "I couldn't turn that into a supported cleaning step. Try being more specific about a column and action."
                    else:
                        updated_df, log = apply_operations(cleaned_df, operations)
                        st.session_state["cleaning_df"] = updated_df
                        reply = "\n".join(f"- {line}" for line in log)
            except Exception as error:
                reply = f"Could not apply that request: {friendly_error_message(error)}"

            st.session_state["cleaning_chat"].append({"role": "assistant", "content": reply})
            st.rerun()

with reader_tab:
    st.write("Upload an image to extract text and tables using OCR.")
    uploaded_image = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "gif", "bmp"])

    if uploaded_image is not None:
        try:
            with st.spinner("Extracting text from image..."):
                markdown_text, df, output_path = process_image_ocr(uploaded_image, "ocr_output.xlsx")

            st.subheader("Extracted text")
            st.markdown(markdown_text)

            if df is not None:
                st.subheader("Extracted table")
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    "Download as Excel",
                    data=(BASE_DIR / "ocr_output.xlsx").read_bytes(),
                    file_name="ocr_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.download_button(
                    "Download as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="ocr_output.csv",
                    mime="text/csv",
                )
            else:
                st.info("No tables were detected in the image. Check the extracted text above.")
        except Exception as error:
            st.error(f"Could not process the image: {friendly_error_message(error)}")
