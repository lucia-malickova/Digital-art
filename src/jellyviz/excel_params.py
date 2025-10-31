import re, math
import numpy as np
import pandas as pd

def _yn_to_num(val):
    if val is None or (isinstance(val, float) and math.isnan(val)): return np.nan
    s = str(val).strip().lower()
    yes = {'si','sí','sim','yes','y','true','verdadero','verdadeiro'}
    no  = {'no','nao','não','false','falso'}
    if s in yes: return 1.0
    if s in no:  return 0.0
    if 'sí' in s or s.startswith('si') or 'sim' in s: return 1.0
    if s == 'no' or 'não' in s or 'nao' in s: return 0.0
    return np.nan

def _prop_from_phrase(val):
    if val is None or (isinstance(val, float) and math.isnan(val)): return np.nan
    s = str(val).strip().lower()
    cues = [
        ('ninguna',0.0),('nula',0.0),('muy baja',0.15),('baja',0.3),
        ('media',0.5),('moderada',0.5),('alta',0.8),('muy alta',0.95),
        ('nenhuma',0.0),('muito baixa',0.15),('baixa',0.3),('média',0.5),('muito alta',0.95)
    ]
    for k,v in cues:
        if k in s: return v
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*%?', s)
    if m:
        x = float(m.group(1).replace(',', '.'))
        return np.clip(x/100.0 if '%' in s else x, 0, 1)
    return np.nan

def _find_col(columns, patterns):
    for col in columns:
        s = str(col).lower()
        if any(p in s for p in patterns):
            return col
    return None

def params_from_excel(path):
    """Return dict(amplitude, frequency, inclusion, hue) derived from Excel."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    df = xl.parse(xl.sheet_names[0]).copy()
    cols = list(df.columns)

    c_has_women     = _find_col(cols, ['¿cuenta con mujeres', 'mujeres dentro', 'mulheres', 'mujeres'])
    c_prop_total    = _find_col(cols, ['proporción de mujeres','proporcao','proporção','total del personal'])
    c_prop_lider    = _find_col(cols, ['liderazgo','liderança','roles de liderazgo','directoras','gerentes'])
    c_has_tech      = _find_col(cols, ['área de tecnología','area de tecnologia','tecnolog'])
    c_prop_techteam = _find_col(cols, ['equipo del área de tecnología','equipe de tecnologia','equipo de tecnologia'])
    c_sector        = _find_col(cols, ['sector industrial','rama de actividad','setor','ramo'])

    def safe(c): return df[c] if (c is not None and c in df.columns) else pd.Series([np.nan]*len(df))
    s_has_women     = safe(c_has_women).map(_yn_to_num)
    s_prop_total    = safe(c_prop_total).map(_prop_from_phrase)
    s_prop_lider    = safe(c_prop_lider).map(_prop_from_phrase)
    s_has_tech      = safe(c_has_tech).map(_yn_to_num)
    s_prop_techteam = safe(c_prop_techteam).map(_prop_from_phrase)

    def fill(series, val):
        if not isinstance(series, pd.Series) or not series.notna().any():
            return pd.Series([val]*len(df))
        return series

    # Fallbacks (so it always runs even if some columns are missing)
    s_has_women     = fill(s_has_women, 1.0)
    s_prop_total    = fill(s_prop_total, 0.45)
    s_prop_lider    = fill(s_prop_lider, 0.25)
    s_has_tech      = fill(s_has_tech, 0.60)
    s_prop_techteam = fill(s_prop_techteam, 0.30)

    amplitude = float(np.nanmean(0.6*s_prop_total + 0.4*s_prop_techteam))
    frequency = float(np.nanmean(0.7*s_prop_lider + 0.3*s_has_tech))
    inclusion = float(np.nanmean(0.3*s_has_women + 0.2*s_has_tech + 0.25*s_prop_total + 0.25*s_prop_techteam))

    if c_sector is not None and c_sector in df.columns:
        hue = (abs(hash(' | '.join(df[c_sector].astype(str).fillna('NA').tolist()))) % 360) / 360.0
    else:
        hue = 0.60

    def safe_val(x, default):
        try:
            if math.isnan(x): return default
        except:
            pass
        return float(np.clip(x, 0, 1))

    return dict(
        amplitude=safe_val(amplitude, 0.5),
        frequency=safe_val(frequency, 0.5),
        inclusion=safe_val(inclusion, 0.5),
        hue=safe_val(hue, 0.60),
    )
