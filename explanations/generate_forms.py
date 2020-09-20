import argparse
import os
import os.path
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient import errors
from googleapiclient.discovery import build


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos')
    parser.add_argument('--app-script-dir', default='forms')
    parser.add_argument('--token-file', default='token.pickle')
    parser.add_argument('--project-name', default='cfrl')
    args = parser.parse_args()
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f'Video directory does not exist: {args.video_dir}')
    return args


# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/script.projects']

SAMPLE_CODE = '''
function helloWorld() {
  console.log("Hello, world!");
}
'''.strip()

SAMPLE_MANIFEST = '''
{
  "timeZone": "America/New_York",
  "exceptionLogging": "CLOUD"
}
'''.strip()


def load_gs_from_folder(path):
    gs_text = {}
    for file_name in os.listdir(path):
        root, ext = os.path.splitext(file_name)
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path) and ext == '.gs':
            gs_text[root] = open(file_path).read().strip()
    return gs_text


def build_requests(gs_text: dict):
    request = {'files': []}
    for name, text in gs_text.items():
        file_req = {
                       'name': name,
                       'type': 'SERVER_JS',
                       'source': text
                   }, {
                       'name': 'appsscript',
                       'type': 'JSON',
                       'source': SAMPLE_MANIFEST
                   }
        request['files'].append(file_req)
    return request


def main():
    """Calls the Apps Script API.
    """

    args = parse_args()
    token_file = args.token_file

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)

    service = build('script', 'v1', credentials=creds)

    # Call the Apps Script API
    try:
        # Create a new project
        request = {'title': args.project_name}
        response = service.projects().create(body=request).execute()
    except errors.HttpError as error:
        # The API encountered a problem.
        print(error.content)

    try:
        # Upload two files to the project
        request = build_requests(load_gs_from_folder(args.app_script_dir))
        # request = {
        #     'files': [{
        #         'name': 'hello',
        #         'type': 'SERVER_JS',
        #         'source': SAMPLE_CODE
        #     }, {
        #         'name': 'appsscript',
        #         'type': 'JSON',
        #         'source': SAMPLE_MANIFEST
        #     }]
        # }
        response = service.projects().updateContent(
            body=request,
            scriptId=response['scriptId']).execute()
        print('https://script.google.com/d/' + response['scriptId'] + '/edit')
    except errors.HttpError as error:
        # The API encountered a problem.
        print(error.content)


if __name__ == '__main__':
    main()
