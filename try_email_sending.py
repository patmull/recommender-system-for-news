import smtplib
import traceback

from mail_sender import send_error_email


def email_sending():
    raise ValueError("Testing exception")


try:
    email_sending()
except smtplib.SMTPException as e:
    send_error_email(traceback.format_exc())
