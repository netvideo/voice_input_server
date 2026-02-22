"""中文数字转阿拉伯数字模块 (ITN)."""

import re

_DIGIT = str.maketrans({
    '零': '0', '〇': '0',
    '一': '1', '壹': '1',
    '二': '2', '贰': '2', '两': '2',
    '三': '3', '叁': '3',
    '四': '4', '肆': '4',
    '五': '5', '伍': '5',
    '六': '6', '陆': '6',
    '七': '7', '柒': '7',
    '八': '8', '捌': '8',
    '九': '9', '玖': '9',
})

def _conv(s):
    r = s.translate(_DIGIT)
    if '十' in s:
        r = r.replace('十', '0')
        if r.startswith('0'):
            r = '1' + r
    for u, v in [('千','000'),('百','00'),('万','0000'),('亿','00000000')]:
        if u in s:
            r = r.replace(u, v)
    return re.sub(r'^0+', '', r) or '0'

def _repl_chapter(m):
    return f"第{_conv(m.group(1))}{m.group(2)}"

def _repl_year(m):
    return f"{_conv(m.group(1))}年"

def _repl_num(m):
    return _conv(m.group(0))

def itn(text, language='zh-CN'):
    if not text:
        return text
    zh_languages = {'zh', 'zh-cn', 'zh-tw', 'chinese'}
    if language and language.lower() not in zh_languages:
        return text
    text = re.sub(r'第([零一二三四五六七八九十百]+)(章|节|页|卷|部|集|篇)', _repl_chapter, text)
    text = re.sub(r'([零一二三四五六七八九十]{2,4})年', _repl_year, text)
    text = re.sub(r'[零一二三四五六七八九十百千万亿]+', _repl_num, text)
    return text

if __name__ == "__main__":
    for t in ["第三章第十五节", "一九八九年", "九十九元", "我有一百二十三个苹果"]:
        print(t, "->", itn(t))
