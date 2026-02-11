from __future__ import annotations
import logging
import smtplib
import os
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Callable


logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, channels: List[str] = None):
        self.channels = channels or ["log"]
        self._handlers: List[Callable[[str, str], None]] = []
        for ch in self.channels:
            if ch == "log":
                self._handlers.append(self._log_alert)
            elif ch == "email":
                self._handlers.append(self._email_alert)

    def _log_alert(self, subject: str, body: str):
        logger.warning(f"[ALERT] {subject}: {body}")

    def _email_alert(self, subject: str, body: str):
        smtp_host = os.getenv("SMTP_HOST", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender    = os.getenv("ALERT_SENDER", "alerts@experiment-vault.io")
        recipient = os.getenv("ALERT_RECIPIENT", "")
        if not recipient:
            return
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = recipient
        try:
            with smtplib.SMTP(smtp_host, smtp_port) as s:
                s.sendmail(sender, [recipient], msg.as_string())
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

    def send(self, subject: str, body: str):
        for h in self._handlers:
            h(subject, body)

    def check_and_alert(self, drift_results: Dict[str, Dict],
                         model_name: str = "unknown"):
        drifted = [k for k, v in drift_results.items() if v.get("drifted")]
        if drifted:
            self.send(
                f"[ExperimentVault] Drift detected: {model_name}",
                f"Features with drift: {', '.join(drifted)}\n\nDetails:\n" +
                "\n".join(f"  {k}: PSI={v.get('psi', 0):.3f}, KS_p={v.get('ks_pvalue', 1):.4f}"
                          for k, v in drift_results.items() if v.get("drifted"))
            )
            return True
        return False
