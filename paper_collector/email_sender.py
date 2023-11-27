from email.message import EmailMessage
import smtplib
from utils import custom_utils as cu

class EmailSender():
    
    def __init__(self):
        self.load_preferences()
        self.set_account()
        self.login()

    def load_preferences(self):
        self.email_account = cu.load_json("./preferences/email_account.json")
    
    def set_account(self):
        self.mail_id = self.email_account['mail_id']
        self.mail_pw = self.email_account['mail_pw']
    
    def login(self):
        self.server = smtplib.SMTP('smtp.gmail.com', 587)
        self.server.starttls()
        self.server.login(self.mail_id, self.mail_pw)

    def make_msg(self, subject, text, receiver):
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self.mail_id
        msg['To'] = receiver
        msg.set_content(text)
        return msg

    def send_msg(self, text, subject='Paper Summarization'):
        
        msg = self.make_msg(subject, text, self.mail_id)
        self.server.send_message(msg)