"""Doi chieu moi con so trong results/*.md voi output dang luu trong notebook.

Chay:  python src/doi_chieu_so.py            # ca ba cap khoi 3, 4, 5
       python src/doi_chieu_so.py 03         # rieng mot khoi

Vi sao co file nay. Khoi 2 tung commit mot bang so ma code hien tai khong tai lap
duoc o bat ky seed nao, va khoi 3 tung de toan bo markdown mo ta sai chinh output
nam ngay duoi no sau khi sua loi hoi tu cua fit_logit. Ca hai deu KHONG lo ra khi
chay lai, vi chay lai chi sinh output moi chu khong doc phan chu. Chung chi lo ra
khi dat hai ben canh nhau, va day la ham lam viec do.

Cach doi chieu. Lay moi so trong output cua notebook lam tap tham chieu, roi voi
moi so kieu Viet Nam (dau phay thap phan) trong file .md, hoi: co gia tri nao trong
tap tham chieu ma lam tron den dung so chu so cua no thi ra chinh no khong. Nho
phep lam tron nay ma "0,0143" van khop voi 0.01428 trong output.

Ket qua khong phai danh sach loi. Mot so hop le van bi bao neu no den tu noi khac:
so cua khoi truoc, hang so bang tra t, so hoc suy ra tai cho, hay con so cu duoc
giu lai co y de doi chieu. Vi vay ham in kem dong van chua no, de nguoi doc phan
xu nhanh thay vi phai di tim.

GIOI HAN, phai biet truoc khi tin no. Tap tham chieu la MOI so trong output, khong
phan biet so do thuoc bang nao. Nen ham nay bat duoc "so khong ton tai" ma KHONG bat
duoc "so co that nhung gan vao menh de sai": mot gia tri nhu 0,00006 xuat hien o
nhieu cho trong output, nen no van pass ke ca khi dang dung o cau cua mot phuong an
khac. Dung la loai loi da xay ra o dong open_credit_lines chieu giam. Cho loai do
khong co cach nao ngoai doc doi chieu bang voi bang.

Cung co y bo qua so nguyen tran (74, 1998, "10 bin"): dua vao thi phan lon canh bao
la so muc, nam trong trich dan va so thu tu, va tieng on do lam ca cong cu mat tac
dung. So co dau cham nghin thi CO bat (1.504, 22.500) vi do la so lieu that.

Nhanh do de lai ba canh bao gia da biet: to_markdown cat so 0 cuoi nen "0.6270" in ra
thanh "0.627" va trong y het mot so nguyen co dau cham nghin. Khong tach duoc neu
khong biet bang nao dang o dinh dang nao, nen cu de no bao roi bo qua.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent.parent
CAP = {
    "03": ("notebooks/03_scorecard.ipynb", "results/scorecard.md"),
    "04": ("notebooks/04_monotone_xgboost.ipynb", "results/model_comparison.md"),
    "05": ("notebooks/05_calibration_psi.ipynb", "results/calibration_psi.md"),
}

# So trong output notebook: dau phay la phan cach nghin (pandas), dau cham la thap phan.
SO_OUTPUT = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?")
# So trong van ban tieng Viet: dau cham phan cach nghin, dau phay thap phan.
# Hai dang duoc bat: co phan thap phan (0,0143 va 1.504,5), va so nguyen co dau cham
# nghin (1.504, 22.500). So nguyen tran bi bo qua co y, xem docstring dau file.
#
# Nhanh thu hai phai co ranh gioi hai dau. Cac bang dan tu to_markdown giu dinh dang
# kieu Anh (dau cham la thap phan), nen khong co ranh gioi thi "171.8246" bi cat ra
# thanh "171.824" va bao nham hang loat.
SO_MARKDOWN = re.compile(r"\d+(?:\.\d{3})*,\d+|(?<![\d.,])\d{1,3}(?:\.\d{3})+(?![\d.,])")


def so_trong_output(duong_dan: Path) -> set:
    """Moi gia tri so xuat hien trong output da luu cua notebook."""
    nb = json.loads(duong_dan.read_text(encoding="utf-8"))
    ra = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                text = "".join(o["text"])
            else:
                text = "".join(o.get("data", {}).get("text/plain", ""))
            for m in SO_OUTPUT.findall(text):
                try:
                    ra.add(abs(float(m.replace(",", ""))))
                except ValueError:
                    pass
    return ra


def khop(x: float, so_le: int, tham_chieu: set) -> bool:
    """x co phai ban lam tron cua mot gia tri nao trong tap tham chieu khong."""
    return any(abs(round(v, so_le) - x) < 1e-12 or abs(v - x) < 1e-12 for v in tham_chieu)


def quet(md: Path, nb: Path) -> list:
    """Tra ve (so dong, con so, dong van) cho moi so khong khop."""
    tham_chieu = so_trong_output(nb)
    ra = []
    for i, dong in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
        for tok in SO_MARKDOWN.findall(dong):
            x = float(tok.replace(".", "").replace(",", "."))
            so_le = len(tok.split(",")[1]) if "," in tok else 0
            if khop(x, so_le, tham_chieu):
                continue
            ra.append((i, tok, dong.strip()[:110]))
    return ra


def main() -> int:
    chon = sys.argv[1:] or list(CAP)
    for khoi in chon:
        nb_ten, md_ten = CAP[khoi]
        la = quet(GOC / md_ten, GOC / nb_ten)
        print(f"\n=== khoi {khoi}: {md_ten} so voi {nb_ten}")
        if not la:
            print("    moi con so deu khop output notebook")
            continue
        print(f"    {len(la)} so khong khop, doc tung dong roi phan xu:")
        for dong, tok, van in la:
            print(f"    dong {dong:>4}  {tok:>12}  |  {van}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
