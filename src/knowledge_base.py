knowledge_base = {
    "reset_password": "Allows users to reset their password. Steps: Go to login page → Click 'Forgot Password' → Enter registered email → Check email for reset link → Create new password → Login again. Common issues: If no email received, check spam folder or verify email. If link expired, request a new reset link. Security tips: Use strong passwords, avoid reuse, never share reset links.",
    "api_limits": "Defines API usage limits. Requests per minute: 60. Tokens per day: 100000. Error 429 means rate limit exceeded; retry after cooldown. Best practices: Implement request throttling, cache frequent responses, monitor API usage, and optimize prompts to reduce token consumption.",
    "billing": "Handles subscriptions, payments, and invoices. Features include viewing invoices, updating payment methods, upgrading or downgrading plans, and cancelling subscriptions. Supported payment methods: Credit Card, Debit Card, UPI. FAQs: Invoices are available under Account → Billing. If payment fails, account may be suspended after a grace period. Refunds are processed within 5–7 business days. Grace period is 3 days after failed payment."
}


def search_kb(query: str) -> str:
    query = query.lower()
    hits = []

    for k, v in knowledge_base.items():
        if query in k or query in v.lower():
            hits.append(f"[{k}] {v}")

    return "/n".join(hits) if hits else "No relevant KB entry found."
