import io, json, os, re, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import fitz
import torch
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).parent

CATALOGS = ROOT / "catalogs"
CATALOGS.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
INDEX=DATA/'index.json'
VECS=DATA/'vectors.npy'

THUMBS=DATA/'thumbs'
THUMBS.mkdir(parents=True, exist_ok=True)
st.set_page_config(page_title='HUB Lighting Finder',page_icon='💡',layout='wide')
st.markdown('<style>.block-container{max-width:1300px}.score{font-size:1.25rem;font-weight:700;color:#f59e0b}</style>',unsafe_allow_html=True)
st.title('💡 HUB Lighting Finder')
st.caption('Αναγνώριση προϊόντων φωτισμού από φωτογραφία και σύγκριση με εταιρικούς καταλόγους PDF')

@st.cache_resource
def model():
    name='openai/clip-vit-base-patch32'
    return CLIPModel.from_pretrained(name),CLIPProcessor.from_pretrained(name)

def embed(im):
    m,p=model(); x=p(images=im.convert('RGB'),return_tensors='pt')
    with torch.no_grad(): v=m.get_image_features(**x)
    v=v/v.norm(dim=-1,keepdim=True); return v.cpu().numpy()[0]

def brand_from_name(n):
    rules={'ARKOS':'ARKOSLIGHT','FLEXA':'FLEXA LIGHTING','AXOLIGHT':'AXOLIGHT','IL FANALE':'IL FANALE','BOVER':'BOVER','9010':'9010','CONTARDI':'CONTARDI','XAL':'XAL','ZAVA':'ZAVA','VESOI':'VESOI','GRAYPANTS':'GRAYPANTS','FARO':'FARO','CATALOGUE NEWS':'VISTOSI'}
    u=n.upper()
    for k,v in rules.items():
        if k in u:return v
    return Path(n).stem

def title_from_text(t):
    lines=[re.sub(r'\s+',' ',x).strip() for x in t.splitlines()]
    lines=[x for x in lines if 2<len(x)<90 and not x.isdigit()]
    return lines[0] if lines else 'Σελίδα προϊόντος'

def ingest_pdf(path, progress=None):
    doc=fitz.open(path); brand=brand_from_name(path.name); rows=[]; vecs=[]
    for i,page in enumerate(doc):
        # Skip very early editorial pages only when no useful text; otherwise index every page.
        text=page.get_text('text')[:2500]
        pix=page.get_pixmap(matrix=fitz.Matrix(1.05,1.05),alpha=False)
        im=Image.open(io.BytesIO(pix.tobytes('jpeg'))).convert('RGB')
        key=hashlib.md5(f'{path.name}:{i}'.encode()).hexdigest()+'.jpg'; target=THUMBS/key; im.save(target,quality=80)
        rows.append({'brand':brand,'catalog':path.name,'page':i+1,'title':title_from_text(text),'text':text,'thumb':str(target.relative_to(ROOT))})
        vecs.append(embed(im))
        if progress: progress.progress((i+1)/len(doc),text=f'{path.name}: σελίδα {i+1}/{len(doc)}')
    return rows,vecs

def rebuild():
    rows=[]; vecs=[]; pdfs=sorted(CATALOGS.glob('*.pdf'))
    if not pdfs:return 0,0
    bar=st.progress(0)
    for p in pdfs:
        r,v=ingest_pdf(p,bar); rows+=r; vecs+=v
    INDEX.write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8'); np.save(VECS,np.asarray(vecs,dtype='float32')); bar.empty()
    return len(pdfs),len(rows)

def load_index():
    if not INDEX.exists() or not VECS.exists():return [],None
    return json.loads(INDEX.read_text(encoding='utf-8')),np.load(VECS)

with st.sidebar:
    st.header('Διαχείριση καταλόγων')
    uploads=st.file_uploader('Πρόσθεσε PDF καταλόγους',type=['pdf'],accept_multiple_files=True)
if uploads and st.button('Αποθήκευση PDF'):
    CATALOGS.mkdir(parents=True, exist_ok=True)

    for f in uploads:
        file_path = CATALOGS / f.name
        file_path.write_bytes(f.getbuffer())

    st.success(f'Αποθηκεύτηκαν {len(uploads)} αρχεία.')
        st.success(f'Αποθηκεύτηκαν {len(uploads)} αρχεία.')
    pdfs=list(CATALOGS.glob('*.pdf')); st.metric('Κατάλογοι',len(pdfs))
    if st.button('Δημιουργία ευρετηρίου',type='primary'):
        n,p=rebuild(); st.success(f'Έτοιμο: {n} κατάλογοι, {p} σελίδες.')
    st.caption('Η πρώτη δημιουργία ευρετηρίου κατεβάζει το μοντέλο CLIP και απαιτεί σύνδεση στο Internet.')

rows,vecs=load_index(); brands=sorted({r['brand'] for r in rows})
query=st.file_uploader('Ανέβασε φωτογραφία φωτιστικού',type=['jpg','jpeg','png','webp'])
c1,c2=st.columns([2,1]); selected=c1.multiselect('Εταιρείες',brands); topk=c2.slider('Αποτελέσματα',3,20,8)
if query:
    im=Image.open(query).convert('RGB'); st.image(im,width=320,caption='Φωτογραφία αναζήτησης')
    if vecs is None: st.warning('Πρόσθεσε καταλόγους και δημιούργησε πρώτα το ευρετήριο.')
    else:
        q=embed(im); sims=vecs@q; ids=[i for i,r in enumerate(rows) if not selected or r['brand'] in selected]; ids=sorted(ids,key=lambda i:sims[i],reverse=True)[:topk]
        st.subheader('Πιθανότερες αντιστοιχίες')
        for rank,i in enumerate(ids,1):
            r=rows[i]; a,b=st.columns([1,2])
            with a: st.image(ROOT/r['thumb'],use_container_width=True)
            with b:
                st.markdown(f"### {rank}. {r['brand']}")
                st.markdown(f"<div class='score'>Ομοιότητα {max(0,float(sims[i])):.1%}</div>",unsafe_allow_html=True)
                st.write(f"**{r['title']}**")
                st.write(f"Κατάλογος: {r['catalog']} · Σελίδα {r['page']}")
                snippet=re.sub(r'\s+',' ',r['text'])[:700]; st.caption(snippet or 'Δεν εξήχθη κείμενο από τη σελίδα.')
            st.divider()
else:
    if rows: st.info(f'Το ευρετήριο είναι έτοιμο με {len(rows):,} σελίδες από {len(set(r["catalog"] for r in rows))} καταλόγους.')
    else: st.info('Ανέβασε PDF καταλόγους από το αριστερό μενού και πάτησε «Δημιουργία ευρετηρίου».')
