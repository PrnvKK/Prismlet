"""The three-question example from the model card."""

from .encoder import _text

STATE = {"ticket": "My card was charged twice for the same order and I still haven't got a refund."}
STATE_TEXT = _text(STATE)

QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "payments, charges and refunds",
                          "tech": "app or website problems",
                          "delivery": "shipping and order status"}},
    "urgent": {"type": "yes_no", "instructions": "Does this need a reply within the hour?"},
    "anger": {"type": "score", "instructions": "How upset is the customer?",
              "criteria": ["calm", "annoyed", "angry", "furious"]},
}
