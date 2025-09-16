# calculator_gui.py
import math
import cmath
import ast
import operator
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from typing import Union, Any
Number = Union[int, float, complex]
# ---------- 常量 ----------
PI = math.pi
E = math.e
# ---------- 全局设置 ----------
USE_DEGREES = False  # 全局角度开关；GUI 会提供切换
# ---------- 工具函数 ----------
def is_close_to_zero(x: float, tol=1e-12) -> bool:
    return abs(x) < tol

def to_radians_if_needed(x: Number) -> Number:
    if USE_DEGREES:
        # 对复数的角度转换：只转换实部作为角度输入比较常见；对复数三角函数直接使用 cmath（以弧度）
        if isinstance(x, complex):
            # treat real part as degrees, imag ignored for conversion
            return complex(math.radians(x.real), x.imag)
        else:
            return math.radians(x)
    return x

def from_radians_if_needed(x: Number) -> Number:
    if USE_DEGREES:
        if isinstance(x, complex):
            return complex(math.degrees(x.real), x.imag)
        else:
            return math.degrees(x)
    return x

def format_number(x: Number) -> str:
    if isinstance(x, complex):
        # format complex nicely
        re = x.real
        im = x.imag
        re_s = f"{re:.12g}" if not is_close_to_zero(re) else "0"
        im_s = f"{abs(im):.12g}"
        sign = "+" if im >= 0 else "-"
        return f"{re_s}{sign}{im_s}j"
    else:
        return f"{x:.12g}"

# ---------- 基本运算（支持复数） ----------
def add(a: Number, b: Number) -> Number:
    return a + b

def subtract(a: Number, b: Number) -> Number:
    return a - b

def multiply(a: Number, b: Number) -> Number:
    return a * b

def divide(a: Number, b: Number) -> Number:
    if b == 0:
        raise ZeroDivisionError("除数不能为0")
    return a / b

def power(a: Number, b: Number) -> Number:
    return a ** b

def nth_root(x: Number, n: int) -> Number:
    if n == 0:
        raise ValueError("开0次方未定义")
    # use complex support via cmath
    if isinstance(x, complex) or x < 0:
        return cmath.exp(cmath.log(x) / n)
    else:
        return x ** (1.0 / n)

def factorial(n: int) -> int:
    if isinstance(n, complex):
        raise ValueError("阶乘不支持复数输入")
    n_int = int(n)
    if n_int != n or n_int < 0:
        raise ValueError("阶乘只对非负整数定义")
    return math.factorial(n_int)

# ---------- 指数与对数 ----------
def exp(x: Number) -> Number:
    return cmath.exp(x) if isinstance(x, complex) else math.exp(x)

def ln(x: Number) -> Number:
    # support complex
    if isinstance(x, complex) or x <= 0:
        return cmath.log(x)
    return math.log(x)

def log10(x: Number) -> Number:
    if isinstance(x, complex) or x <= 0:
        return cmath.log10(x)
    return math.log10(x)

def log_base(x: Number, base: Number) -> Number:
    return ln(x) / ln(base)

# ---------- 三角与反三角（支持复数 via cmath） ----------
def sin(x: Number) -> Number:
    xr = to_radians_if_needed(x)
    return cmath.sin(xr) if isinstance(xr, complex) else math.sin(xr)

def cos(x: Number) -> Number:
    xr = to_radians_if_needed(x)
    return cmath.cos(xr) if isinstance(xr, complex) else math.cos(xr)

def tan(x: Number) -> Number:
    xr = to_radians_if_needed(x)
    try:
        return cmath.tan(xr) if isinstance(xr, complex) else math.tan(xr)
    except Exception:
        return cmath.tan(xr)

def asin(x: Number) -> Number:
    # result expressed in current angle unit
    res = cmath.asin(x) if isinstance(x, complex) or x < -1 or x > 1 else math.asin(x)
    return from_radians_if_needed(res)

def acos(x: Number) -> Number:
    res = cmath.acos(x) if isinstance(x, complex) or x < -1 or x > 1 else math.acos(x)
    return from_radians_if_needed(res)

def atan(x: Number) -> Number:
    res = cmath.atan(x) if isinstance(x, complex) else math.atan(x)
    return from_radians_if_needed(res)

def atan2(y: Number, x: Number) -> Number:
    # cmath does not have atan2; use math for real inputs
    if isinstance(x, complex) or isinstance(y, complex):
        return cmath.phase(complex(x, y))
    return from_radians_if_needed(math.atan2(y, x))

# ---------- 双曲函数 ----------
def sinh(x: Number) -> Number:
    return cmath.sinh(x) if isinstance(x, complex) else math.sinh(x)

def cosh(x: Number) -> Number:
    return cmath.cosh(x) if isinstance(x, complex) else math.cosh(x)

def tanh(x: Number) -> Number:
    return cmath.tanh(x) if isinstance(x, complex) else math.tanh(x)

def asinh(x: Number) -> Number:
    return cmath.asinh(x) if isinstance(x, complex) else math.asinh(x)

def acosh(x: Number) -> Number:
    return cmath.acosh(x) if isinstance(x, complex) else math.acosh(x)

def atanh(x: Number) -> Number:
    return cmath.atanh(x) if isinstance(x, complex) else math.atanh(x)

# ---------- 安全表达式解析 ----------
# 支持数字、复数字面量（用 j）、名称（pi,e）、函数调用（来自 safe_names）、一元/二元操作
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

# 可用名与函数映射
def _wrap_real_or_complex(fn):
    def wrapper(x, *args):
        return fn(x, *args)
    return wrapper

SAFE_NAMES = {
    'pi': PI,
    'e': E,
    'i': complex(0,1),
    'j': complex(0,1),
    # functions
    'sin': sin,
    'cos': cos,
    'tan': tan,
    'asin': asin,
    'acos': acos,
    'atan': atan,
    'atan2': atan2,
    'sinh': sinh,
    'cosh': cosh,
    'tanh': tanh,
    'asinh': asinh,
    'acosh': acosh,
    'atanh': atanh,
    'sqrt': lambda x: nth_root(x, 2),
    'root': nth_root,
    'ln': ln,
    'log': log_base,
    'log10': log10,
    'exp': exp,
    'fact': factorial,
    'factorial': factorial,
    'abs': abs,
    're': lambda z: z.real if isinstance(z, complex) else float(z),
    'im': lambda z: z.imag if isinstance(z, complex) else 0.0,
    # constants
    'PI': PI,
    'E': E,
}

def eval_expr(expr: str) -> Number:
    """
    使用 ast 安全解析并计算表达式。
    支持函数调用、数字（含复数 j）、变量名（来自 SAFE_NAMES）。
    """
    node = ast.parse(expr, mode='eval')
    return _eval_ast(node.body)

def _eval_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Num):  # Python <3.8
        return node.n
    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError("不支持的二元运算")
    if isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError("不支持的一元运算")
    if isinstance(node, ast.Name):
        if node.id in SAFE_NAMES:
            return SAFE_NAMES[node.id]
        raise ValueError(f"未知名称: {node.id}")
    if isinstance(node, ast.Call):
        func = _eval_ast(node.func)
        args = [_eval_ast(a) for a in node.args]
        # only allow calling functions from SAFE_NAMES map
        if callable(func):
            return func(*args)
        raise ValueError("不允许调用该对象")
    if isinstance(node, ast.Attribute):
        raise ValueError("属性访问不允许")
    if isinstance(node, ast.Subscript):
        raise ValueError("下标访问不允许")
    raise ValueError("不支持的表达式元素")

# ---------- GUI (Tkinter) ----------
class CalculatorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("增强计算器")
        self.geometry("480x640")
        self.resizable(False, False)
        self.history = []

        self._build_ui()

    def _build_ui(self):
        # 顶部输入
        frm_top = ttk.Frame(self)
        frm_top.pack(fill='x', padx=8, pady=8)

        self.entry_var = tk.StringVar()
        self.entry = ttk.Entry(frm_top, textvariable=self.entry_var, font=("Segoe UI", 14))
        self.entry.pack(fill='x', side='left', expand=True)
        self.entry.bind("<Return>", lambda e: self._on_eval())

        btn_eval = ttk.Button(frm_top, text="=", width=4, command=self._on_eval)
        btn_eval.pack(side='left', padx=4)

        # 角度切换
        self.angle_var = tk.StringVar(value="RAD")
        chk = ttk.Checkbutton(frm_top, text="Degrees", command=self._toggle_degrees, variable=self.angle_var, onvalue="DEG", offvalue="RAD")
        # We'll keep storing as on/off via the variable, but we'll update global flag in callback
        chk.pack(side='left', padx=4)

        # 结果显示
        frm_res = ttk.Frame(self)
        frm_res.pack(fill='x', padx=8)
        self.result_var = tk.StringVar(value="")
        lbl_res = ttk.Label(frm_res, textvariable=self.result_var, font=("Segoe UI", 16), anchor='e', background="white", relief='sunken')
        lbl_res.pack(fill='x', pady=8)

        # 按钮区
        frm_buttons = ttk.Frame(self)
        frm_buttons.pack(padx=8, pady=6, fill='both', expand=True)

        btn_texts = [
            ['7','8','9','/','sqrt','('],
            ['4','5','6','*','pow',')'],
            ['1','2','3','-','ln','log'],
            ['0','.','+','%','exp','fact'],
            ['pi','e','i','Ans','deg','AC'],
            ['sin','cos','tan','asin','acos','atan'],
            ['sinh','cosh','tanh','abs','re','im'],
            ['hist','clr','copy','paste','','']
        ]

        for r, row in enumerate(btn_texts):
            frm_row = ttk.Frame(frm_buttons)
            frm_row.pack(fill='x', expand=True)
            for c, txt in enumerate(row):
                if not txt:
                    lbl = ttk.Label(frm_row, text="", width=6)
                    lbl.pack(side='left', padx=2, pady=2)
                    continue
                btn = ttk.Button(frm_row, text=txt, width=6)
                btn.pack(side='left', padx=2, pady=2)
                btn.config(command=lambda t=txt: self._on_button(t))

        # 历史显示区
        frm_hist = ttk.Frame(self)
        frm_hist.pack(fill='both', padx=8, pady=6, expand=True)
        lbl_hist = ttk.Label(frm_hist, text="History", anchor='w')
        lbl_hist.pack(fill='x')
        self.listbox = tk.Listbox(frm_hist, height=6)
        self.listbox.pack(fill='both', expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self._use_history())

    def _toggle_degrees(self):
        global USE_DEGREES
        # toggle based on current string value
        val = self.angle_var.get()
        if val == "DEG":
            USE_DEGREES = True
        else:
            USE_DEGREES = False

    def _on_button(self, txt: str):
        txt_low = txt.lower()
        if txt in ('AC',):
            self.entry_var.set("")
            self.result_var.set("")
            return
        if txt == 'Ans':
            if self.history:
                self.entry_var.set(self.entry_var.get() + str(self.history[-1][1]))
            return
        if txt == 'hist':
            self._show_history()
            return
        if txt == 'clr':
            self.listbox.delete(0, tk.END)
            self.history.clear()
            return
        if txt == 'copy':
            self.clipboard_clear()
            self.clipboard_append(self.result_var.get())
            return
        if txt == 'paste':
            try:
                txtp = self.clipboard_get()
                self.entry_var.set(self.entry_var.get() + txtp)
            except Exception:
                pass
            return
        # insert function or character
        if txt in ('sqrt','pow','ln','log','exp','fact','sin','cos','tan','asin','acos','atan','sinh','cosh','tanh','abs','re','im'):
            self.entry_var.set(self.entry_var.get() + txt + "(")
            return
        if txt in ('pi','e','i'):
            self.entry_var.set(self.entry_var.get() + txt)
            return
        if txt == '%':
            self.entry_var.set(self.entry_var.get() + "/100")
            return
        # degrees toggle quick button (alias)
        if txt == 'deg':
            self.angle_var.set("DEG" if self.angle_var.get()!="DEG" else "RAD")
            self._toggle_degrees()
            return

        # default: append character
        self.entry_var.set(self.entry_var.get() + txt)

    def _on_eval(self):
        expr = self.entry_var.get().strip()
        if not expr:
            return
        try:
            result = eval_expr(expr)
            # format for display
            res_str = format_number(result)
            self.result_var.set(res_str)
            # save history (expr, raw result)
            self.history.append((expr, result))
            self.listbox.insert(tk.END, f"{expr} = {res_str}")
        except Exception as e:
            messagebox.showerror("计算错误", str(e))

    def _show_history(self):
        # just focus history listbox
        self.listbox.focus_set()

    def _use_history(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        expr, res = self.history[idx]
        self.entry_var.set(expr)
        self.result_var.set(format_number(res))
def run():
    app = CalculatorGUI()
    app.mainloop()
if __name__ == "__main__":
    run()
