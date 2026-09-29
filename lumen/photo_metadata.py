"""Capture metadata used by borders, including metadata embedded in AI copies."""
import json
import math
import re
from pathlib import Path

FIELDS = ('date', 'body', 'lens', 'aperture', 'shutter', 'iso', 'focal', 'brand', 'lens_brand')


def clean(record):
    if not isinstance(record, dict): return {}
    return {key: re.sub(r'[\x00-\x1f]', ' ', str(record.get(key, '')))[:200] for key in FIELDS}


def from_tags(tags):
    def text(key):
        value = tags.get(key, '')
        return str(value).strip() if value is not None else ''
    def numeric(key):
        try:
            value = float(tags.get(key, 0))
            return value if math.isfinite(value) and value > 0 else None
        except (ValueError, TypeError): return None
    date = text('DateTimeOriginal') or text('CreateDate')
    if re.match(r'^\d{4}:\d{2}:\d{2}', date): date = date[:10].replace(':', '-') + date[10:]
    aperture, exposure, focal, iso = (numeric(k) for k in ('FNumber','ExposureTime','FocalLength','ISO'))
    shutter = ''
    if exposure:
        reciprocal=1/exposure
        shutter=(f'1/{round(reciprocal)} s' if exposure<1 and abs(reciprocal-round(reciprocal))<.02*reciprocal else f'{exposure:.3g} s')
    lens = text('LensModel') or text('Lens')
    if not lens and isinstance(tags.get('LensID'), str): lens = tags['LensID']
    brand = text('Make')
    if brand.upper().startswith('NIKON'):brand='Nikon'
    lens_brand = text('LensMake')
    if not lens_brand:
        identity=lens
        if isinstance(tags.get('LensID'),str) and ' or ' not in tags['LensID'].lower():identity+=' '+tags['LensID']
        for name in ('SIGMA','TAMRON','ZEISS','VOIGTLANDER','SAMYANG','VILTROX','CANON','NIKON','NIKKOR','FUJIFILM','FUJINON','PANASONIC','LUMIX','SONY','LEICA'):
            if name in identity.upper(): lens_brand = name; break
    return clean(dict(date=date,body=text('Model'),lens=lens,brand=brand,lens_brand=lens_brand,
                      aperture=f'f/{aperture:g}' if aperture else '',shutter=shutter,
                      iso=f'ISO {iso:g}' if iso else '',focal=f'{focal:g} mm' if focal else ''))


def description(photo, provenance=None):
    return json.dumps({'application':'Lumen photo','version':1,'photo':clean(photo),
                       'processing':provenance or {}},ensure_ascii=True)


def embedded(path):
    if Path(path).suffix.lower() not in ('.dng','.tif','.tiff'): return {}
    try:
        import tifffile
        with tifffile.TiffFile(path) as f:
            text = f.pages[0].description or ''
            if len(text)>16384: return {}
            data = json.loads(text)
            if data.get('application')=='Lumen photo' and data.get('version')==1:
                return clean(data.get('photo'))
    except (OSError, ValueError, AttributeError): pass
    return {}
