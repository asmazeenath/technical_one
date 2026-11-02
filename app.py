import os
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai

# ---------------- CONFIGURATION ----------------
st.set_page_config(
    page_title="Retail AI Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB = Path(__file__).resolve().parent / "retail_ai.db"
MODEL = "gemini-3.5-flash"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: rgba(128,128,128,.08);
    padding: 16px; border-radius: 12px;
}
.stButton button {border-radius: 9px;}
</style>
""", unsafe_allow_html=True)

st.title("🛍️ Retail AI Intelligence Platform")
st.caption("Gemini AI • Customer Intelligence • Transaction Monitoring • Model Observability")

# ---------------- DATABASE ----------------
def load_data():
    if not DB.exists():
        return None, None, None

    with sqlite3.connect(DB) as conn:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table'",
            conn
        )["name"].tolist()

        required = {"customer_churn", "transaction_anomalies"}
        if not required.issubset(tables):
            return None, None, tables

        customers = pd.read_sql_query(
            'SELECT * FROM "customer_churn"', conn
        )
        transactions = pd.read_sql_query(
            'SELECT * FROM "transaction_anomalies"', conn
        )

        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_reviews (
                transaction_id TEXT PRIMARY KEY,
                review_status TEXT NOT NULL DEFAULT 'Needs review',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        reviews = pd.read_sql_query(
            "SELECT * FROM anomaly_reviews", conn
        )

    return customers, transactions, tables


loaded = load_data()

if loaded[0] is None:
    st.error(
        "Database or required tables are missing. "
        "Run your notebook and save customer_churn and "
        "transaction_anomalies to retail_ai.db first."
    )
    st.stop()

customers, transactions, tables = loaded

if customers.empty or transactions.empty:
    st.warning("One or more tables are empty. Check your notebook output.")

if "churn" not in customers.columns:
    st.error("The customer_churn table needs a 'churn' column.")
    st.stop()

if "anomaly" not in transactions.columns:
    st.error("The transaction_anomalies table needs an 'anomaly' column.")
    st.stop()

# ---------------- DATA NORMALIZATION ----------------
def truthy_churn(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "churn", "churned"}:
            return True
        if v in {"0", "false", "no", "active", "not churned"}:
            return False
        return None
    if value in (0, 1):
        return bool(value)
    return None


def is_unusual(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {
            "unusual", "anomaly", "anomalous", "outlier",
            "fraud", "suspicious", "-1"
        }
    try:
        return float(value) == -1
    except (ValueError, TypeError):
        return False


customers = customers.copy().reset_index(drop=True)
transactions = transactions.copy().reset_index(drop=True)

customers["_churn_bool"] = customers["churn"].map(truthy_churn)
transactions["_unusual_bool"] = transactions["anomaly"].map(is_unusual)

# Discover useful identifiers from actual column names.
def find_column(df, candidates):
    lookup = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]
    return None


customer_id_col = find_column(
    customers, ["customer_id", "customerid", "customer_code", "customer"]
)
transaction_id_col = find_column(
    transactions, ["transaction_id", "transactionid", "invoice_no",
                   "invoice", "order_id", "orderid"]
)
name_col = find_column(
    customers, ["customer_name", "name", "full_name", "customername"]
)
inactive_col = find_column(
    customers, ["days_inactive", "days_since_last_purchase",
                "recency_days", "inactive_days"]
)

# Stable fallback IDs for display and reviews.
customers["_display_id"] = (
    customers[customer_id_col].astype(str)
    if customer_id_col else
    ["Customer row " + str(i + 1) for i in range(len(customers))]
)
transactions["_display_id"] = (
    transactions[transaction_id_col].astype(str)
    if transaction_id_col else
    ["Transaction row " + str(i + 1) for i in range(len(transactions))]
)

# Reviews are matched by a transaction's actual ID when available.
transactions["_review_key"] = transactions["_display_id"].astype(str)

with sqlite3.connect(DB) as conn:
    reviews = pd.read_sql_query(
        "SELECT * FROM anomaly_reviews", conn
    )

review_map = {}
if not reviews.empty:
    review_map = dict(zip(
        reviews["transaction_id"].astype(str),
        reviews["review_status"]
    ))

transactions["review_status"] = transactions["_review_key"].map(
    review_map
).astype("str").fillna("Needs review")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Platform Controls")

   

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "AI Assistant",
            "Customer Churn",
            "Transaction Intelligence",
            "Data Quality",
            "Model Monitoring",
        ]
    )

    st.divider()
    st.write("**Database status**")
    st.success("Connected to SQLite")
    st.write(f"Customer records: {len(customers):,}")
    st.write(f"Transaction records: {len(transactions):,}")
    st.caption("AI answers use the loaded database records.")

# ---------------- SHARED FUNCTIONS ----------------
def safe_records(df, limit=30):
    """Return a compact, bounded preview without internal helper columns."""
    internal = [
        "_churn_bool", "_unusual_bool", "_display_id", "_review_key",
        "review_status"
    ]
    cols = [c for c in df.columns if c not in internal]
    return df[cols].head(limit).to_dict(orient="records")


def metric_status(value):
    result = truthy_churn(value)
    if result is True:
        return "CHURN LABEL: YES"
    if result is False:
        return "CHURN LABEL: NO"
    return "CHURN LABEL: UNKNOWN"


def customer_lookup(query):
    """Exact identifier match first; otherwise search actual field values."""
    q = str(query).strip()
    if not q:
        return customers.iloc[0:0]

    if customer_id_col:
        exact = customers[
            customers[customer_id_col].astype(str).str.lower() == q.lower()
        ]
        if not exact.empty:
            return exact

    searchable = [
        c for c in customers.columns
        if not str(c).startswith("_")
        and (
            customers[c].dtype == "object"
            or pd.api.types.is_numeric_dtype(customers[c])
        )
    ]

    mask = pd.Series(False, index=customers.index)
    for col in searchable:
        mask |= customers[col].astype(str).str.contains(
            re.escape(q), case=False, na=False
        )
    return customers[mask].head(30)


def transaction_lookup(query):
    q = str(query).strip()
    if not q:
        return transactions.iloc[0:0]

    if transaction_id_col:
        exact = transactions[
            transactions[transaction_id_col].astype(str).str.lower()
            == q.lower()
        ]
        if not exact.empty:
            return exact

    searchable = [
        c for c in transactions.columns if not str(c).startswith("_")
    ]
    mask = pd.Series(False, index=transactions.index)
    for col in searchable:
        mask |= transactions[col].astype(str).str.contains(
            re.escape(q), case=False, na=False
        )
    return transactions[mask].head(30)

api_key= os.getenv("GEMINI_API_KEY")
client= genai. Client(api_key=api_key)
def gemini_answer(question, context):
    if not api_key:
        return (
            "Gemini is not configured. Set GEMINI_API_KEY in your "
            "environment or Streamlit secrets, then restart the app."
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL,
            contents=f"""
You are the AI analyst for a retail intelligence platform.

Rules:
1. Answer using only the supplied database evidence.
2. Never invent customer IDs, names, amounts, dates, predictions or metrics.
3. Clearly distinguish a stored churn label from a proven future outcome.
4. Explain anomaly flags as signals for review, not proof of fraud.
5. If evidence is missing or ambiguous, explicitly say so.
6. For specific records, repeat exact values from the evidence.
7. Give a concise conclusion, supporting evidence, and next steps.
8. Do not claim that you queried a database; the evidence was retrieved
   by the application.

DATABASE EVIDENCE:
{context}

USER QUESTION:
{question}
""",
        )
        return response.text or "Gemini returned an empty response."
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return "Gemini API quota/rate limit reached. Check your quota and retry."
        if "403" in msg or "API_KEY_INVALID" in msg:
            return "Gemini API key rejected. Check your key and API access."
        return f"Gemini request failed: {msg[:500]}"


def churn_text(row):
    status = row.get("_churn_bool")
    if status is True:
        return "Yes — stored churn label is 1/true."
    if status is False:
        return "No — stored churn label is 0/false."
    return f"Unknown — stored value: {row.get('churn')}"


def download_csv(df, filename):
    visible = df.loc[:, ~df.columns.astype(str).str.startswith("_")]
    st.download_button(
        "⬇️ Download CSV",
        data=visible.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.subheader("Executive Overview")

    total_churn = int((customers["_churn_bool"] == True).sum())
    known_churn = int(customers["_churn_bool"].notna().sum())
    churn_rate = (
        total_churn / known_churn * 100 if known_churn else 0
    )

    a, b, c, d = st.columns(4)
    a.metric("Customers", f"{len(customers):,}")
    b.metric("Churn labelled YES", f"{total_churn:,}")
    c.metric("Known-label churn rate", f"{churn_rate:.2f}%")
    d.metric(
        "Unusual transactions",
        f"{int(transactions['_unusual_bool'].sum()):,}"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")
        chart = customers["_churn_bool"].map(
            {True: "Churn: Yes", False: "Churn: No"}
        ).fillna("Unknown").value_counts()
        st.bar_chart(chart)

    with right:
        st.subheader("Transaction Classification")
        st.bar_chart(
            transactions["_unusual_bool"].map(
                {True: "Unusual", False: "Not flagged"}
            ).value_counts()
        )

    st.subheader("Recent Unusual Transactions")
    st.dataframe(
        transactions[transactions["_unusual_bool"]].drop(
            columns=["_unusual_bool", "_display_id", "_review_key"],
            errors="ignore"
        ).head(10),
        width="stretch",
        hide_index=True,
    )

    st.info(
        "The churn rate describes available labels. It is not a validated "
        "future-churn probability unless your notebook trained and evaluated "
        "a predictive model."
    )

# ---------------- AI ASSISTANT ----------------
elif page == "AI Assistant":
    st.subheader("💬 Gemini Retail Analyst")
    st.write(
        "Ask questions about customers, churn, transactions, sales, or data quality. "
        "Customer and transaction details are retrieved from your SQLite database."
    )

    examples = [
        "Is customer 12346 churned?",
        "Show details for customer 12357 ",
        "Is transaction  stockCode 85048 unusual?",
        "List all unusual transactions",
        "Which customers have been inactive the longest?",
        "Summarize the current churn and anomaly situation",
    ]
    if "user_question" not in st.session_state:
     st.session_state.user_question = ""

    def update_question():
     selected = st.session_state.example_question

     if selected != "Choose an example...":
        st.session_state.user_question = selected

    st.selectbox(
    "Example questions",
    ["Choose an example..."] + examples,
    key="example_question",
    on_change=update_question
)

    question = st.text_area(
    "Ask your question",
    key="user_question",
    placeholder="Select an example or type your own question",
    height=100
)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col1, col2 = st.columns([1, 5])

    with col1:
        ask = st.button(
            "Ask Gemini",
            type="primary",
            width="stretch",
        )

    with col2:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            q = question.strip()
            qlow = q.lower()

            evidence = {
                "customer_count": len(customers),
                "transaction_count": len(transactions),
                "known_churn_labels": int(
                    customers["_churn_bool"].notna().sum()
                ),
                "churn_yes": int(
                    (customers["_churn_bool"] == True).sum()
                ),
                "unusual_transaction_count": int(
                    transactions["_unusual_bool"].sum()
                ),
            }

            customer_results = customers.iloc[0:0].copy()
            transaction_results = transactions.iloc[0:0].copy()

            # 1. Extract customer IDs, including CUST-102 and CUST102.
            id_match = re.search(
                r"\b(?:customer|cust|client)"
                r"\s*(?:id)?\s*[:#]?\s*"
                r"([A-Za-z][A-Za-z0-9_-]*|\d+)\b",
                q,
                flags=re.I,
            )

            # Also recognize a standalone ID such as CUST-102.
            generic_id = re.search(
                r"\b(?:CUST|CUSTOMER|CLIENT)[-_]?[A-Z0-9]+\b",
                q,
                flags=re.I,
            )

            customer_id = (
                id_match.group(1) if id_match else
                generic_id.group(0) if generic_id else None
            )

            # 2. Extract transaction IDs, including TXN-501.
            txn_match = re.search(
                r"\b(?:transaction|txn|invoice)"
                r"\s*(?:id)?\s*[:#]?\s*"
                r"([A-Za-z][A-Za-z0-9_-]*|\d+)\b",
                q,
                flags=re.I,
            )

            generic_txn = re.search(
                r"\b(?:TXN|TRANS|INV)[-_]?[A-Z0-9]+\b",
                q,
                flags=re.I,
            )

            transaction_id = (
                txn_match.group(1) if txn_match else
                generic_txn.group(0) if generic_txn else None
            )

            # 3. Exact database lookup first.
            if customer_id:
                customer_results = customer_lookup(customer_id)

            if transaction_id:
                transaction_results = transaction_lookup(transaction_id)

            # 4. Aggregate transaction questions.
            if any(word in qlow for word in [
                "all unusual",
                "list unusual",
                "suspicious transactions",
                "unusual transactions",
                "flagged transactions",
                "show anomalies",
            ]):
                transaction_results = transactions[
                    transactions["_unusual_bool"]
                ].head(50).copy()

            # 5. Inactive customer questions.
            if any(word in qlow for word in [
                "inactive",
                "longest",
                "days inactive",
                "at risk",
            ]):
                if inactive_col:
                    customer_results = customers.sort_values(
                        inactive_col,
                        ascending=False,
                        na_position="last",
                    ).head(20).copy()
                elif not customer_id:
                    evidence["inactive_days_note"] = (
                        "No recognized inactive-days column exists. "
                        "Cannot rank customers by inactivity."
                    )

            # 6. Collect exact customer records.
            customer_details = []

            for _, row in customer_results.iterrows():
                details = {
                    str(k): (
                        None if pd.isna(v) else str(v)
                    )
                    for k, v in row.items()
                    if not str(k).startswith("_")
                }

                details["verified_churn_interpretation"] = churn_text(row)
                customer_details.append(details)

            # 7. Collect exact transaction records.
            transaction_details = []

            for _, row in transaction_results.iterrows():
                details = {
                    str(k): (
                        None if pd.isna(v) else str(v)
                    )
                    for k, v in row.items()
                    if not str(k).startswith("_")
                }

                details["application_anomaly_flag"] = (
                    "Unusual"
                    if row["_unusual_bool"]
                    else "Not flagged"
                )

                transaction_details.append(details)

            evidence["matched_customers"] = customer_details
            evidence["matched_transactions"] = transaction_details

            # 8. Handle missing exact records without inventing results.
            if customer_id and customer_results.empty:
                evidence["customer_lookup_result"] = (
                    f"No customer matched ID '{customer_id}'. "
                    "Do not infer their churn status."
                )

            if transaction_id and transaction_results.empty:
                evidence["transaction_lookup_result"] = (
                    f"No transaction matched ID '{transaction_id}'. "
                    "Do not infer its anomaly status."
                )

            # 9. Handle direct churn questions using database evidence.
            if customer_id and not customer_results.empty:
                if "churn" in customers.columns:
                    statuses = [
                        churn_text(row)
                        for _, row in customer_results.iterrows()
                    ]
                    evidence["exact_churn_results"] = statuses

            # 10. Ask Gemini to explain retrieved evidence.
            with st.spinner(
                "Searching database and consulting Gemini..."
            ):
                if (
                    customer_id
                    and customer_results.empty
                    and not transaction_id
                ):
                    answer = (
                        f"### Customer not found\n\n"
                        f"No matching record for **{customer_id}** "
                        "was found in the customer table. "
                        "I cannot determine their churn status. "
                        "Check the customer ID column and stored IDs."
                    )

                elif (
                    transaction_id
                    and transaction_results.empty
                    and customer_results.empty
                ):
                    answer = (
                        f"### Transaction not found\n\n"
                        f"No matching record for **{transaction_id}** "
                        "was found in the transaction table. "
                        "Check the transaction ID column and stored IDs."
                    )

                elif not api_key:
                    answer = (
                        "Gemini API key is missing. Configure "
                        "`GEMINI_API_KEY` in your environment or "
                        "Streamlit secrets to enable AI explanations.\n\n"
                        "The retrieved records are available below."
                    )

                else:
                    answer = gemini_answer(q, evidence)

            st.session_state.chat_history.append({
                "question": q,
                "answer": answer,
                "evidence": evidence,
            })

    # 11. Display conversation and evidence.
    for item in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            with st.expander("View retrieved database evidence"):
                st.json(item["evidence"])
# ---------------- CUSTOMER CHURN ----------------
elif page == "Customer Churn":
    st.subheader("👥 Customer Intelligence")

    if customer_id_col:
        st.caption(f"Customer identifier: {customer_id_col}")

    search = st.text_input(
        "Search customer by ID, name or another field",
        placeholder="Enter a customer ID or name"
    )

    view = customers.copy()
    if search:
        view = customer_lookup(search)

    f1, f2 = st.columns(2)
    with f1:
        churn_filter = st.selectbox(
            "Churn status",
            ["All", "Churned", "Not churned", "Unknown"]
        )
    with f2:
        sort_by_inactive = st.checkbox(
            "Sort by days inactive", value=False
        )

    if churn_filter == "Churned":
        view = view[view["_churn_bool"] == True]
    elif churn_filter == "Not churned":
        view = view[view["_churn_bool"] == False]
    elif churn_filter == "Unknown":
        view = view[view["_churn_bool"].isna()]

    if sort_by_inactive and inactive_col:
        view = view.sort_values(
            inactive_col, ascending=False, na_position="last"
        )

    st.metric("Matching customers", len(view))
    st.dataframe(
        view.drop(
            columns=["_churn_bool", "_display_id"], errors="ignore"
        ),
        width="stretch",
        hide_index=True
    )
    download_csv(view, "customer_churn_report.csv")

    st.subheader("Exact Customer Lookup")
    customer_options = view["_display_id"].astype(str).tolist()

    if customer_options:
        chosen = st.selectbox("Select a customer", customer_options)
        row = view[view["_display_id"].astype(str) == chosen].iloc[0]

        st.write("**Stored churn status:**", churn_text(row))
        c1, c2, c3 = st.columns(3)
        c1.metric("Customer ID", str(row["_display_id"]))
        c2.metric(
            "Days inactive",
            str(row[inactive_col]) if inactive_col else "Not available"
        )
        c3.metric(
            "Churn field value",
            str(row["churn"])
        )

        st.json({
            str(k): (None if pd.isna(v) else str(v))
            for k, v in row.items()
            if not str(k).startswith("_")
        })
    else:
        st.info("No matching customers found.")

# ---------------- TRANSACTION INTELLIGENCE ----------------
elif page == "Transaction Intelligence":
    st.subheader("🔍 Transaction Intelligence")

    search_txn = st.text_input(
        "Search transaction",
        placeholder="Enter transaction ID or amount"
    )

    show = transactions.copy()
    if search_txn:
        show = transaction_lookup(search_txn)

    flag_filter = st.selectbox(
        "Classification",
        ["All", "Unusual only", "Not flagged"]
    )

    if flag_filter == "Unusual only":
        show = show[show["_unusual_bool"]]
    elif flag_filter == "Not flagged":
        show = show[~show["_unusual_bool"]]

    st.metric("Matching transactions", len(show))
    st.dataframe(
        show.drop(
            columns=["_unusual_bool", "_display_id", "_review_key"],
            errors="ignore"
        ),
        width="stretch",
        hide_index=True
    )
    download_csv(show, "transaction_intelligence.csv")

    st.subheader("Review flagged transactions")
    flagged = show[show["_unusual_bool"]].copy()

    if not flagged.empty:
        review_choices = [
            "Needs review",
            "Reviewed - legitimate",
            "Investigate further"
        ]

        edit_cols = [
            c for c in [
                "_display_id", "Quantity", "Price",
                "transaction_value", "anomaly", "review_status"
            ] if c in flagged.columns
        ]

        edit_df = flagged[edit_cols].copy().rename(
            columns={"_display_id": "transaction_id"}
        )

        edited = st.data_editor(
            edit_df,
            width="stretch",
            hide_index=True,
            disabled=[
                c for c in edit_df.columns if c != "review_status"
            ],
            column_config={
                "review_status": st.column_config.SelectboxColumn(
                    "Review status",
                    options=review_choices,
                    required=True
                )
            }
        )

        if st.button("Save Review Changes", type="primary"):
            try:
                with sqlite3.connect(DB) as conn:
                    conn.executemany("""
                        INSERT INTO anomaly_reviews
                            (transaction_id, review_status, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(transaction_id) DO UPDATE SET
                            review_status=excluded.review_status,
                            updated_at=CURRENT_TIMESTAMP
                    """, edited[
                        ["transaction_id", "review_status"]
                    ].astype(str).values.tolist())

                st.success("Review statuses saved to SQLite.")
                st.rerun()
            except sqlite3.Error as e:
                st.error(f"Could not save reviews: {e}")
    else:
        st.success("No unusual transactions in this filtered view.")

    st.warning(
        "An anomaly is a statistical flag, not proof of fraud. "
        "Verify transaction details before taking action."
    )

# ---------------- DATA QUALITY ----------------
elif page == "Data Quality":
    st.subheader("🧪 Data Quality Report")

    for name, df in [
        ("Customer Churn", customers),
        ("Transaction Anomalies", transactions),
    ]:
        with st.expander(f"{name} — quality checks", expanded=True):
            report = pd.DataFrame({
                "Data type": df.dtypes.astype(str),
                "Missing values": df.isna().sum(),
                "Missing %": (df.isna().mean() * 100).round(2),
                "Unique values": df.nunique(dropna=True),
            })
            st.dataframe(report, use_container_width=True)
            st.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Dataset Summary")
    st.write("Tables detected:", ", ".join(tables))
    st.write("Customer ID column:", customer_id_col or "Not detected")
    st.write("Transaction ID column:", transaction_id_col or "Not detected")
    st.write("Inactive-days column:", inactive_col or "Not detected")

    st.info(
        "This dashboard checks the saved model tables. For full source-data "
        "validation, also persist your notebook's original CSV/SQL checks "
        "for dates, quantities, categories, duplicate keys and invalid values."
    )

# ---------------- MODEL MONITORING ----------------
elif page == "Model Monitoring":
    st.subheader("📈 Model Monitoring")

    a, b, c = st.columns(3)
    a.metric("Customer rows", len(customers))
    b.metric("Transaction rows", len(transactions))
    c.metric(
        "Unknown churn labels",
        int(customers["_churn_bool"].isna().sum())
    )

    st.subheader("Anomaly review progress")
    flagged_count = int(transactions["_unusual_bool"].sum())
    reviewed_count = int(
        transactions.loc[
            transactions["_unusual_bool"], "review_status"
        ].isin(["Reviewed - legitimate", "Investigate further"]).sum()
    )

    m1, m2 = st.columns(2)
    m1.metric("Flagged", flagged_count)
    m2.metric("Reviewed", reviewed_count)

    if flagged_count:
        st.progress(min(reviewed_count / flagged_count, 1.0))

    st.subheader("Data availability")
    st.write({
        "customer_identifier": customer_id_col or "Not detected",
        "transaction_identifier": transaction_id_col or "Not detected",
        "inactive_days": inactive_col or "Not detected",
        "customer_rows": len(customers),
        "transaction_rows": len(transactions),
        "unknown_churn_labels": int(customers["_churn_bool"].isna().sum()),
        "missing_transaction_values": int(
            transactions.isna().sum().sum()
        ),
    })

    st.warning(
        "Production-grade monitoring still requires persisted prediction "
        "timestamps, model versions, historical metrics, feature drift, "
        "prediction drift and scheduled ground-truth evaluation. "
        "This page does not claim those checks are already implemented."
    )

st.divider()
st.caption(
    "Retail AI Intelligence • Gemini explanations grounded in saved database "
    "records • Validate predictions before operational decisions"
)