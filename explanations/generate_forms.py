import argparse
import os
import os.path
import pickle
import json

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient import errors
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-dir', default='videos', help='directory to load the videos from')
    parser.add_argument('--app-script-dir', default='forms', help='directory for the app script gs files')
    parser.add_argument('--token-file', default='token.pickle', help='file to save authentication credentials')
    parser.add_argument('--project-name', default='cfrl', help='name google api project')
    parser.add_argument('--deployment-folder-id', default='1OEO0oGm8eSr2P7Vn3LMYuk64SIOmnBwA',
                        help='deployment folder google id')
    args = parser.parse_args()
    # if not os.path.exists(args.video_dir):
    #     raise FileNotFoundError(f'Video directory does not exist: {args.video_dir}')
    return args


# If modifying these scopes, delete the file token.pickle.
SCOPES = [
    'https://www.googleapis.com/auth/script.projects',
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/drive.scripts',
    'https://www.googleapis.com/auth/script.external_request',
]


def auth(token_file):
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
    return creds


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


def build_requests(gs_text: dict, config: dict):
    request = {'files': []}
    for name, text in gs_text.items():
        file_req = {
            'name': name,
            'type': 'SERVER_JS',
            'source': text
        }
        request['files'].append(file_req)
    request['files'].append({
        'name': 'appsscript',
        'type': 'JSON',
        'source': SAMPLE_MANIFEST
    })
    request['files'].append({
        'name': 'config',
        'type': 'SERVER_JS',
        'source': build_config(config)
    })
    return request


def build_config(config: dict):
    json_dict = json.dumps(config)
    text = f'var CONFIG = {json_dict};'
    return text


def create_folder(drive_service, name='videos'):
    file_metadata = {
        'name': name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    file = drive_service.files().create(body=file_metadata,
                                        fields='id').execute()
    return file


def upload_videos(path, drive_service, folder_id):
    if os.path.exists(path):
        for video in os.listdir(path):
            video_path = os.path.join(path, video)
            file_metadata = {'name': video}
            media = MediaFileUpload('files/photo.jpg', mimetype='image/jpeg')


def main():
    """Calls the Apps Script API.
    """

    args = parse_args()

    credentials = auth(args.token_file)

    drive_service = build('drive', 'v3', credentials=credentials)
    script_service = build('script', 'v1', credentials=credentials)

    upload_folder_id = args.deployment_folder_id
    video_dir = args.video_dir

    # find the video folder and create new one if necessary
    result = drive_service.files().list(
        q=f"'{upload_folder_id}' in parents and name = '{video_dir}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false").execute()
    if len(result['files']) == 1:
        print('found videos folder')
        video_folder_id = result['files'][0]['id']
        print(f"The id is {video_folder_id}")
    else:
        print('creating videos folder')
        file_metadata = {
            'name': f'{video_dir}',
            'parents': [upload_folder_id],
            'mimeType': 'application/vnd.google-apps.folder',
        }
        file = drive_service.files().create(body=file_metadata, fields='id').execute()
        video_folder_id = file.get('id')

    find_video_list_result = drive_service.files().list(
        q=f"'{video_folder_id}' in parents and mimeType = 'video/mp4' and trashed = false"
    ).execute()
    if len(find_video_list_result['files']) == 0:
        # upload files to video path
        absolute_file_paths = [os.path.abspath(os.path.join(video_dir, f)) for f in os.listdir(video_dir)]
        for file_path in absolute_file_paths:
            file_metadata = {
                'name': os.path.basename(file_path),
                'parents': [video_folder_id]
            }
            media = MediaFileUpload(file_path,
                                    mimetype='video/mp4',
                                    resumable=True)
            file = drive_service.files().create(body=file_metadata,
                                                media_body=media,
                                                fields='id').execute()
            print(f'uploaded {file_path} with id {file.get("id")}')
    else:
        print(f'videos already uploaded to drive folder {video_dir} with id {video_folder_id}')

    # Call the Apps Script API
    script_id = None
    if script_id is None:
        # Create a new project
        request = {'title': args.project_name}
        response = script_service.projects().create(body=request).execute()
        script_id = response['scriptId']
        print('created new project')
        print(response)

    # TODO: define config
    config = {
        'video_folder_id': video_folder_id,
        'form_name': 'form',
    }

    # upload gs files to appscript project
    request = build_requests(load_gs_from_folder(args.app_script_dir), config)
    response = script_service.projects().updateContent(
        body=request,
        scriptId=script_id).execute()
    print('uploaded gs files')
    print(response)
    print(f"https://script.google.com/d/{response['scriptId']}/edit")

    # # try to deploy script
    # request = {"function": "main", "devMode": True}
    # response = script_service.scripts().run(body=request,
    #                                         scriptId=script_id).execute()
    # print(response)


if __name__ == '__main__':
    main()
