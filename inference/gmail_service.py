import os.path
import base64
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import settings

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

class GmailService:
    def __init__(self):
        self.creds = None
        self.service = None
        self.authenticate()

    def authenticate(self):
        """Shows basic usage of the Gmail API.
        Lists the user's Gmail labels.
        """
        if os.path.exists(settings.GMAIL_TOKEN_FILE):
            self.creds = Credentials.from_authorized_user_file(settings.GMAIL_TOKEN_FILE, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    settings.GMAIL_CREDENTIALS_FILE, SCOPES
                )
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(settings.GMAIL_TOKEN_FILE, "w") as token:
                token.write(self.creds.to_json())

        try:
            self.service = build("gmail", "v1", credentials=self.creds)
        except HttpError as error:
            print(f"An error occurred: {error}")

    def fetch_unread_emails(self, max_results=5):
        """Fetches the latest unread emails."""
        if not self.service:
            return []
            
        emails = []
        try:
            results = self.service.users().messages().list(userId='me', labelIds=['INBOX', 'UNREAD'], maxResults=max_results).execute()
            messages = results.get('messages', [])

            for msg in messages:
                message = self.service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                
                payload = message.get('payload', {})
                headers = payload.get('headers', [])
                
                subject = next((header['value'] for header in headers if header['name'] == 'Subject'), "No Subject")
                sender = next((header['value'] for header in headers if header['name'] == 'From'), "Unknown Sender")
                
                # Extract body
                body = ""
                if 'parts' in payload:
                    for part in payload['parts']:
                        if part['mimeType'] == 'text/plain':
                            if 'data' in part['body']:
                                data = part['body']['data']
                                body = base64.urlsafe_b64decode(data).decode('utf-8')
                                break
                elif 'body' in payload and 'data' in payload['body']:
                    data = payload['body']['data']
                    body = base64.urlsafe_b64decode(data).decode('utf-8')
                
                emails.append({
                    'id': msg['id'],
                    'subject': subject,
                    'sender': sender,
                    'body': body.strip()
                })
        except HttpError as error:
            print(f"An error occurred fetching emails: {error}")
            
        return emails

    def mark_as_read(self, msg_id):
        """Marks an email as read by removing the UNREAD label."""
        if not self.service:
            return
        try:
            self.service.users().messages().modify(
                userId='me', 
                id=msg_id, 
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
        except HttpError as error:
            print(f"An error occurred marking email as read: {error}")
