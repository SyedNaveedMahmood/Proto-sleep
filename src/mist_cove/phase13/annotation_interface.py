"""Local-only blinded review service. Real review requires approved humans."""
import hashlib
import hmac
import json
import re
import time
import copy
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from .annotation_schema import blind_case,validate_submission
from .provenance import canonical,write_new


class ReviewStore:
    def __init__(self,cases,reviewers,directory,domain='GENERATED',protocol_identity=None):
        if domain not in ('GENERATED','REAL_TRAIN'):raise PermissionError('unknown domain')
        if not reviewers:raise PermissionError('no approved qualified reviewers')
        for alias,reviewer in reviewers.items():
            if not re.fullmatch('[A-Za-z0-9_-]{1,64}',alias):raise PermissionError('opaque safe reviewer handle required')
            if reviewer.get('role')!='reviewer':raise PermissionError('reviewer role required')
            if domain=='REAL_TRAIN' and (not reviewer.get('qualified_approved') or reviewer.get('scope')!='PHASE13_TRAIN_LOCAL' or not reviewer.get('credential_category')):
                raise PermissionError('qualified local access approval missing')
        self.cases=copy.deepcopy(cases);self.reviewers=copy.deepcopy(reviewers);self.directory=Path(directory);self.domain=domain
        self.protocol_identity=copy.deepcopy(protocol_identity)
        self.packet_identity=hashlib.sha256(canonical({k:self.case(k) for k in sorted(self.cases)})).hexdigest()
        self.directory.mkdir(parents=True,exist_ok=True);self.locked=set();self.records=[];self.chain='0'*64;self.mutex=RLock()
        self.log=self.directory/'annotation_events.jsonl'
        if self.log.exists():
            if self.log.is_symlink():raise PermissionError('annotation log symlink')
            for line in self.log.read_text().splitlines():
                event=json.loads(line);digest=event.pop('sha256')
                if event['domain']!=self.domain or event['packet_sha256']!=self.packet_identity or event.get('protocol_identity')!=self.protocol_identity:raise ValueError('replayed packet/protocol identity differs')
                if event['previous']!=self.chain or hashlib.sha256(canonical(event)).hexdigest()!=digest:raise ValueError('annotation log chain differs')
                self.chain=digest;self.records.append(event)
                if event['action']=='lock':self.locked.add(event['reviewer'])

    def authenticate(self,token):
        digest=hashlib.sha256(token.encode()).hexdigest()
        for alias,record in self.reviewers.items():
            if hmac.compare_digest(digest,record['token_sha256']):return alias
        raise PermissionError('reviewer credential required')

    def case(self,case_id):
        value=self.cases[case_id]
        return blind_case(value['master'],value['wave_uV'],value.get('context_uV'))

    def append(self,reviewer,action,value):
        with self.mutex:
            if reviewer in self.locked:raise PermissionError('review export already locked')
            if reviewer not in self.reviewers:raise PermissionError('unknown reviewer')
            if action=='assessment':value=validate_submission(value,self.cases)
            elif action!='lock':raise ValueError('unknown action')
            event={'previous':self.chain,'reviewer':reviewer,'domain':self.domain,'action':action,'packet_sha256':self.packet_identity,
                   'UTC_seconds':time.time(),'payload':value,'protocol_identity':self.protocol_identity}
            digest=hashlib.sha256(canonical(event)).hexdigest()
            with self.log.open('a') as stream:
                stream.write(json.dumps({**event,'sha256':digest},allow_nan=False)+'\n');stream.flush()
            self.chain=digest;self.records.append(event)
            if action=='lock':
                self.locked.add(reviewer)
                latest={v['payload']['case_id']:v['payload'] for v in self.records if v['reviewer']==reviewer and v['action']=='assessment'}
                export={'domain':self.domain,'reviewer':reviewer,'locked':True,
                        'qualified_approved':self.reviewers[reviewer].get('qualified_approved',False),
                        'case_set_sha256':hashlib.sha256(canonical(sorted(self.cases))).hexdigest(),
                        'final_chain_sha256':digest,'packet_sha256':self.packet_identity,'assessments':latest,'protocol_identity':self.protocol_identity}
                write_new(self.directory/(reviewer+'_locked.json'),export)
            return digest


def handler_for(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def respond(self,status,value,content_type='application/json'):
            data=json.dumps(value,allow_nan=False).encode() if content_type=='application/json' else value.encode()
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'self'; connect-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'; form-action 'none'")
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
        def authorized(self):
            host=self.headers.get('Host','')
            if host not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):raise PermissionError('local Host required')
            origin=self.headers.get('Origin')
            if origin and origin!='http://'+host:raise PermissionError('same-origin required')
            bearer=self.headers.get('Authorization','')
            if not bearer.startswith('Bearer '):raise PermissionError('credential required')
            return store.authenticate(bearer[7:])
        def do_GET(self):
            if self.path in ('/','/app.js'):
                name='annotation_ui.html' if self.path=='/' else 'annotation_ui.js'
                self.respond(200,Path(__file__).with_name(name).read_text(),'text/html; charset=utf-8' if self.path=='/' else 'application/javascript');return
            try:
                self.authorized()
                if self.path=='/api/cases':self.respond(200,{'cases':list(store.cases),'domain':store.domain})
                elif self.path.startswith('/api/case/'):
                    case_id=self.path[len('/api/case/'):]
                    self.respond(200,store.case(case_id))
                else:self.respond(404,{'error':'unknown endpoint'})
            except PermissionError:self.respond(403,{'error':'local authorized reviewer required'})
            except KeyError:self.respond(404,{'error':'unknown case'})
        def do_POST(self):
            try:
                reviewer=self.authorized();length=int(self.headers.get('Content-Length',0))
                if not 0<length<=65536:raise ValueError('bounded JSON required')
                value=json.loads(self.rfile.read(length))
                action={'/api/assessment':'assessment','/api/lock':'lock'}.get(self.path)
                if action is None:self.respond(404,{'error':'unknown endpoint'});return
                digest=store.append(reviewer,action,value);self.respond(200,{'record_sha256':digest})
            except PermissionError:self.respond(403,{'error':'authorization or locked-review boundary'})
            except (ValueError,KeyError,TypeError):self.respond(400,{'error':'invalid annotation submission'})
    return Handler


def serve(store,port=8765):
    server=ThreadingHTTPServer(('127.0.0.1',port),handler_for(store))
    print('Local blinded review listening on127.0.0.1; credentials and outputs stay private',flush=True)
    try:server.serve_forever()
    finally:server.server_close()
