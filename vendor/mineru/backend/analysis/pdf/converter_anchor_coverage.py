"""Source-marker coverage for mixed question OCR; this does not verify body semantics."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Any
from mineru_vl_utils.vlm_client.base_client import RequestError

@dataclass
class CoverageResult:
    blocks_list: list
    checked: int
    repaired: list[dict[str,Any]]


def marker_coverage(text: str, question: int | None, options: list[str]) -> bool:
    # Mathematical (A) is not an option label. Ignore recognized math spans.
    visible=re.sub(r'\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$', '', text)
    qmarks=[int(x) for x in re.findall(r'[（(]\s*(\d{1,3})\s*[）)]',visible)]
    omarks=re.findall(r'[（(]\s*([ABCD])\s*[）)]',visible)
    if question is not None and question not in qmarks:return False
    cursor=0
    for label in options:
        try:cursor=omarks.index(label,cursor)+1
        except ValueError:return False
    return True


def _union(boxes):
    return [min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]


def _iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0


def _metadata(block,metas,width,height):
    if '_converter_anchor_evidence' in block:
        return [block['_converter_anchor_evidence']]
    sources=block.get('_converter_joint_sources')
    if sources is not None:
        if any(not isinstance(i,int) or not 0<=i<len(metas) for i in sources):
            raise RequestError('Source anchor ownership is invalid')
        return [metas[i] for i in sources]
    box=[v*(width if i%2==0 else height) for i,v in enumerate(block['bbox'])]
    scored=[(_iou(box,m['roi_bbox_px']),m) for m in metas if 'roi_bbox_px' in m]
    if not scored:return []
    score,meta=max(scored,key=lambda item:item[0])
    return [meta] if score>=.3 else []


def repair_anchor_coverage(predictor,images,extracted_blocks_list,anchor_pages,*,margin_px=12) -> CoverageResult:
    """Check observed source题号/ABCD; missing markers get one source recrop pass.

    OCR metadata contains printed markers and line boxes only. No recognized OCR
    body or inferred math is copied to output. Ambiguous multi-question regions
    are skipped here. The caller should separately log ambiguity, and must not
    describe this limited coverage check as semantic/character accuracy proof.
    """
    if len(images)!=len(extracted_blocks_list) or len(images)!=len(anchor_pages):
        raise ValueError('Page images, extracted blocks and anchor metadata must align')
    work=[];crops=[];checked=0
    for page_idx,(image,blocks,metas) in enumerate(zip(images,extracted_blocks_list,anchor_pages)):
        w,h=image.size
        for block in blocks:
            if block.get('type') not in {'text','vertical_text'}:continue
            evidence=_metadata(block,metas,w,h)
            if any(m.get('ambiguous') for m in evidence):continue
            questions=list(dict.fromkeys(m.get('question') for m in evidence if m.get('question') is not None))
            if len(questions)>1:continue
            question=questions[0] if questions else None
            options=[x for m in evidence for x in m.get('options',[])]
            if question is None and not options:continue
            checked+=1
            text=block.get('content') or ''
            if marker_coverage(text,question,options):continue
            printed=[a for m in evidence for a in m.get('anchors',[])]
            qlines=[a['bbox_px'] for a in printed if a['kind']=='question' and a['value']==question]
            olines=[a['bbox_px'] for a in printed if a['kind']=='option' and a['value'] in options]
            original=[v*(w if i%2==0 else h) for i,v in enumerate(block.get('_converter_unpadded_bbox',block['bbox']))]
            source=_union([original,*qlines,*olines])
            x0,y0,x1,y1=source
            boxes=[]
            if qlines and olines and max(b[3] for b in qlines)<min(b[1] for b in olines):
                # Whitespace between the last stem marker line and first option
                # establishes a geometric boundary; do not duplicate either part.
                first_option=min(b[1] for b in olines)
                stem_lines=[box for meta in evidence for box in meta.get('line_boxes_px',[]) if box[3] < first_option]
                last_stem=max((box[3] for box in stem_lines),default=max(b[3] for b in qlines))
                gap=first_option-last_stem
                boundary=(last_stem+first_option)/2 if stem_lines else first_option-min(margin_px,gap/3)
                boxes=[[max(0,x0-margin_px),max(0,y0-margin_px),min(w,x1+margin_px),boundary],
                       [max(0,x0-margin_px),boundary,min(w,x1+margin_px),min(h,y1+margin_px)]]
            else:
                # An option-only crop keeps its whole source context. Never guess
                # individual glyph coordinates inside a merged OCR line.
                boxes=[[max(0,x0-margin_px),max(0,y0-margin_px),min(w,x1+margin_px),min(h,y1+margin_px)]]
            # Outer expansion may not absorb another independently recognized ROI.
            for box in boxes:
                for other in blocks:
                    if other is block:continue
                    ob=[v*(w if i%2==0 else h) for i,v in enumerate(other['bbox'])]
                    if min(box[2],ob[2])>max(box[0],ob[0]) and min(box[3],ob[3])>max(box[1],ob[1]):
                        box[0]=max(box[0],source[0]);box[1]=max(box[1],source[1])
                        box[2]=min(box[2],source[2]);box[3]=min(box[3],source[3])
            begin=len(crops)
            crops.extend(image.crop(tuple(round(v) for v in box)) for box in boxes)
            work.append({'page':page_idx,'block':block,'question':question,'options':options,
                         'boxes_px':boxes,'start':begin,'end':len(crops)})
    repairs=[]
    try:
        contents=predictor.batch_content_extract(crops,types='text') if crops else []
        if len(contents)!=len(crops):raise RequestError('Recrop output count does not match source crops')
        for item in work:
            repaired='\n\n'.join(str(t) for t in contents[item['start']:item['end']] if t)
            if not marker_coverage(repaired,item['question'],item['options']):
                raise RequestError(f"Source marker coverage still failed after recrop: page {item['page']+1}, question {item['question']}, options {item['options']}")
            item['block']['content']=repaired
            repairs.append({k:v for k,v in item.items() if k not in {'block','start','end'}})
    finally:
        for crop in crops:crop.close()
    return CoverageResult(extracted_blocks_list,checked,repairs)
