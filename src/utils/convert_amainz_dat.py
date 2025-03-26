import pandas as pd
import re

# Ersetze 'input_file.xlsx' durch den Pfad zu deiner Excel-Datei
input_file = '../../data/20240813-FibrosisDB(302_Patients).xlsx'
output_file = '../../data/20240813-FibrosisDB_converted.xlsx'

# Mapping der Spaltennamen aus Liste 1 auf Liste 2
column_mapping = {
    'ID': 'ID',
    'age': 'Age',
    'alat': 'ALAT (U/I)',
    'asat': 'ASAT (U/I)',
    'bilirubin': 'Bilrubin gesamt (mg/dl)',
    'thrombozyten': 'Thrombozyten (Mrd/l)',
    'mcv': 'MCV (fl)',
    'quick': 'Quick (%)',
    'inr': 'INR',
    'leukozyten': 'Leukozyten (Mrd/l)',
    'ptt': 'PTT (sek)',
    'igg': 'IgG (g/l)',
    'albumin': 'Albumin (g/l)',
    'hba1c': 'HbA1c (%)',
    'ap': 'AP (U/I)',
    'harnstoff': 'Harnstoff',
    'hb': 'Hb (g/dl)',
    'kalium': 'Kalium',
    'ggt': 'GGT (U/I)',
    'kreatinin': 'Kreatinin (mg/dl)',
    'gfr': 'GRF (berechnet) (ml/min)',
    'Fibrosen-grad': 'Micro'  # Für den Fall, dass es diese Spalte gibt
}


# Funktion zur Bereinigung der Fibrose-Grad-Einträge
def clean_fibrosis_grade(value):
    if pd.isna(value):
        return value

    # Wenn es einen '-' gibt, zähle solche Einträge
    if '-' in value:
        return None  # Setze auf None oder ein anderes Platzhalterwert, um anzuzeigen, dass es entfernt werden soll

    # Entferne das 'F' und 'F' gefolgt von Zahlen
    value = re.sub(r'^F', '', str(value))

    try:
        # Konvertiere den Wert zu Integer, wenn möglich
        return int(value)
    except ValueError:
        # Falls es keinen numerischen Wert gibt, gib None zurück
        return None


# Excel-Datei einlesen
df = pd.read_excel(input_file)

# Spaltennamen umbenennen
df.rename(columns=column_mapping, inplace=True)

# Bereinige die 'Micro'-Spalte und entferne Zeilen mit None
df['Micro'] = df['Micro'].apply(clean_fibrosis_grade)

# Speichere die Anzahl der entfernten Einträge mit '-'
num_removed_entries = df[df['Micro'].isna()].shape[0]
print(f"Anzahl der Patienten mit '-' im Fibrosegrad, die entfernt wurden: {num_removed_entries}")

# Entferne Zeilen mit None-Werten in der 'Micro'-Spalte
df = df[df['Micro'].notna()]

# Neue Excel-Datei speichern
df.to_excel(output_file, index=False)

print(f"Die Spaltennamen wurden erfolgreich umbenannt und die Datei wurde als '{output_file}' gespeichert.")
