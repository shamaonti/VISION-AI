import smtplib
import ssl
from email.message import EmailMessage
import os

SENDER = 'shamaonti2@gmail.com'
RECEIVER = 'shamaonti18@gmail.com'
APP_PASSWORD = 'ioxv zdup ijwc mfzb'  # Your Gmail app password

def send_email_alert(subject, body, attachments=None, receiver=RECEIVER):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SENDER
    msg['To'] = receiver
    msg.set_content(body)

    if attachments:
        for path in attachments:
            if path and os.path.exists(path):
                with open(path, 'rb') as f:
                    data = f.read()
                    ext = os.path.splitext(path)[1].lower()
                    subtype = 'jpeg' if ext in ['.jpg', '.jpeg'] else 'png'
                    name = os.path.basename(path)
                    msg.add_attachment(data, maintype='image', subtype=subtype, filename=name)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(SENDER, APP_PASSWORD)
        smtp.send_message(msg)
