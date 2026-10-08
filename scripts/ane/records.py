"""A handful of varied Clef records (text only) for comparing the ANE backbone with the PyTorch one."""

RECORDS = [
    {"id": "invoice", "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue",
        "notes": "Customer disputes the second line item and asked for a call back on Friday."}},
     "questions": {
        "status": {"type": "choice", "instructions": "What is the invoice status?",
                   "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."}},
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "This week", "Today"]}}},
    {"id": "ticket", "state": "Our checkout started returning errors an hour ago and orders are blocked. Customers are emailing us screenshots of a 500 page. Engineering has not responded in the on-call channel.",
     "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle the message?",
                       "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": "New business", "hr": "People and hiring"}},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Right now"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
        "angry": {"type": "noul", "instructions": "Is the writer angry at the company?"}}},
    {"id": "security", "state": {"alert": {"source": "edr", "host": "build-07", "process": "powershell.exe -enc JABzAD0ATgBlAHcALQBPAGI=",
        "parent": "winword.exe", "user": "svc-build", "time": "03:12 UTC", "network": "outbound 443 to 203.0.113.9"}},
     "questions": {
        "severity": {"type": "choice", "instructions": "How severe is this alert?",
                     "criteria": {"low": "Benign or expected activity", "medium": "Suspicious, needs review", "high": "Likely compromise", "critical": "Active attack in progress"}},
        "isolate": {"type": "noul", "instructions": "Should the host be isolated from the network now?"},
        "known_good": {"type": "noul", "instructions": "Is this a known-good administrative task?"}}},
    {"id": "review", "state": "I bought this blender three weeks ago. It is loud, the lid cracked on the second use, and customer service took ten days to answer. The smoothies it makes are fine though.",
     "questions": {
        "sentiment": {"type": "score", "instructions": "Overall sentiment of the review.", "criteria": ["Very negative", "Negative", "Mixed", "Positive", "Very positive"]},
        "mentions_defect": {"type": "noul", "instructions": "Does the reviewer report a product defect?"},
        "topic": {"type": "choice", "instructions": "Main topic of the complaint.", "criteria": {"noise": "Loudness", "durability": "Parts breaking", "support": "Customer service", "price": "Cost"}}}},
    {"id": "contract", "state": "Section 9.2: Either party may terminate this Agreement for convenience upon ninety (90) days' prior written notice. Upon termination, all fees paid are non-refundable, and the Customer shall remain liable for all amounts accrued through the effective date of termination. Section 9.3: The Provider may suspend the Services immediately if the Customer is more than thirty (30) days overdue.",
     "questions": {
        "terminate_convenience": {"type": "noul", "instructions": "Can the customer terminate without cause?"},
        "refund": {"type": "choice", "instructions": "What happens to prepaid fees on termination?", "criteria": {"full": "Fully refunded", "pro_rata": "Refunded pro rata", "none": "Not refunded"}},
        "notice_days": {"type": "score", "instructions": "How long is the notice period?", "criteria": ["Under 30 days", "30 to 59 days", "60 to 89 days", "90 days or more"]},
        "suspension": {"type": "noul", "instructions": "May the provider suspend service for late payment?"}}},
    {"id": "travel", "state": {"request": "Book me a train from Zurich to Milan next Tuesday morning, window seat, and a hotel near the station for two nights. Budget is tight, around 150 CHF per night.",
        "profile": {"loyalty": "gold", "language": "de", "company": "Helvetia Tools"}},
     "questions": {
        "intent": {"type": "choice", "instructions": "Primary intent.", "criteria": {"book_travel": "Book transport or lodging", "cancel": "Cancel a booking", "support": "Ask a question", "expense": "File an expense"}},
        "needs_hotel": {"type": "noul", "instructions": "Does the user need accommodation?"},
        "budget": {"type": "score", "instructions": "How constrained is the budget?", "criteria": ["Unconstrained", "Flexible", "Moderate", "Tight", "Very tight"]},
        "multi_leg": {"type": "noul", "instructions": "Does the trip involve more than one destination?"}}},
]


def long_record(characters: int) -> dict:
    """A record whose state is the first `characters` characters of real source code (a long, non-synthetic document)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    sources = [root / "hf-meta" / "joint_schema_model.py", root / "models" / "clef-flash-4bit" / "clef_mlx.py",
               *sorted((root / "scripts" / "ane").glob("*.py"))]
    text = "\n\n".join(p.read_text() for p in sources)
    return {"id": f"code_{characters}", "state": text[:characters], "questions": {
        "language": {"type": "choice", "instructions": "Which programming language is the document written in?",
                     "criteria": {"python": "Python source code", "rust": "Rust source code", "go": "Go source code", "prose": "Plain English text"}},
        "neural": {"type": "noul", "instructions": "Does the document define neural network layers?"},
        "web": {"type": "noul", "instructions": "Does the document implement a web server?"},
        "size": {"type": "score", "instructions": "How large is the codebase shown?", "criteria": ["A snippet", "A small module", "Several modules", "A large project"]}}}
