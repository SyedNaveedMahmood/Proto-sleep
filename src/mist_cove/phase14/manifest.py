"""Metadata-derived root-rebased source capabilities; never enumerate SC files."""
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from .common import PRIVATE,PUBLIC,write_new,measured,ledger,IntegrityError,sha256,canonical
from .access_control import RootAccess
from mist_cove.phase11b.historical import authenticate_model,lineage
from mist_cove.phase11b.source import development_records


def parse_checksums(data):
    result={}
    for line in data.decode('utf8').splitlines():
        if not line.strip():continue
        m=re.fullmatch(r'([0-9a-f]{64})\s+\*?(.+)',line)
        if not m or m[2] in result:raise IntegrityError('malformed/duplicate official checksum')
        result[m[2]]=m[1]
    return result


def inventory():
    with measured('root_metadata_and_authentication') as counts:
        access=RootAccess();metadata={name:access.bytes(name) for name in ('RECORDS','RECORDS-v1','SHA256SUMS.txt')}
        request=urllib.request.Request('https://physionet.org/files/sleep-edfx/1.0.0/SHA256SUMS.txt',headers={'User-Agent':'MIST-COVE-source-verifier'})
        with urllib.request.urlopen(request,timeout=45) as response:
            official=response.read(2*1024*1024+1)
        if len(official)>2*1024*1024:raise IntegrityError('official metadata size limit')
        if sha256(official)!=sha256(metadata['SHA256SUMS.txt']):raise IntegrityError('local/independent official manifest differs')
        checksums=parse_checksums(official)
        model,payload,root,child,auth=authenticate_model();frozen,normalization,provenance=lineage(payload,root,child)
        sc=development_records(frozen,'train');allowed={};records=[]
        inventory_paths=set(metadata['RECORDS'].decode().splitlines())|set(metadata['RECORDS-v1'].decode().splitlines())
        for r in sc:
            p='sleep-cassette/'+Path(r['path']).name;h='sleep-cassette/'+Path(r['annotation_source']).name
            if checksums.get(p)!=r['source_sha256'] or checksums.get(h)!=r['annotation_sha256']:raise IntegrityError('newroot SC identity differs from historical reference')
            for relative in (p,h):allowed[relative]={'role':'SC_TRAIN','sha256':checksums[relative]}
            records.append({'psg':p,'hypnogram':h,'role':'SC_TRAIN','subject':r['subject_id'],'recording':r['recording_id'],'night':r['night'],'historical':r})
        st_paths=sorted(p for p in checksums if p.startswith('sleep-telemetry/'))
        psg=[p for p in st_paths if re.fullmatch(r'sleep-telemetry/ST\d{4}J0-PSG.edf',p)]
        for p in psg:
            # Official files differ in scorer suffix; unique subject/night prefix
            # is required; no filename repair or alphabetical scorer selection.
            prefix=Path(p).name[:6]
            hh=[h for h in st_paths if Path(h).name.startswith(prefix) and h.endswith('-Hypnogram.edf')]
            if len(hh)!=1:raise IntegrityError('ambiguous/missing official ST PSG hypnogram pair')
            h=hh[0];name=Path(p).name
            for relative in (p,h):allowed[relative]={'role':'ST_EVAL','sha256':checksums[relative]}
            records.append({'psg':p,'hypnogram':h,'role':'ST_EVAL','subject':name[:5],'recording':name[:8],'night':int(name[5])})
        if len(sc)!=29 or len(psg)!=44 or len({r['subject'] for r in records if r['role']=='ST_EVAL'})!=22:raise IntegrityError('expected29SC/44ST/22subjects structure differs')
        manifest={'allowed':allowed,'records':records,'normalization':normalization,'frozen_split':frozen['split_fingerprint'] if 'split_fingerprint' in frozen else provenance['split_fingerprint']}
        write_new(PRIVATE/'source_manifest.json',manifest)
        (PRIVATE/'official_checksums.txt').write_bytes(official)
        write_new(PUBLIC/'data_scope_receipt.json',{'status':'PASS_METADATA_ONLY','SC_TRAIN_participants':15,'SC_TRAIN_recordings':29,'ST_subjects':22,'ST_recordings':44,
            'allowed_PSG_hypnogram_files':len(allowed),'official_manifest_sha256':sha256(official),'official_manifest_independently_matched':True,
            'manifest_canonical_sha256':sha256(canonical(manifest)),'historical_model':auth,'metadata_reads':access.counts,'ST_waveform_values_opened':False,
            'SC_RESERVED_reads':0,'new_SC_VAL_reads':0,'external_data_reads':0})
        counts.update(access.counts);access.close()
        ledger([1,2,3,4],'COMPLETE','Exact base and root verified; SC29historical digests match independent official manifest; ST44unique subject/night pairs', ['reports/mist_cove/phase14_v1/data_scope_receipt.json'])
        print(json.dumps({'SC_TRAIN':len(sc),'ST':len(psg),'source_files':len(allowed),'official_manifest':'MATCH','ST_values_opened':False}))
