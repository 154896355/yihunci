# -*- coding: utf-8 -*-
"""云端批量导入流水线（GitHub Actions 运行）
输入: cloud/words.txt（待导入词语，分隔符同前端） + cloud/lib.json（词库快照 [{id,type,words:[{term,...}]}]）
输出: cloud/result.json（导入清单 [{type,words:[{term,meaning,keyPoint,example}],matchTargetGroupId}]）
环境变量: ZHIPU_API_KEY
逻辑与前端 v4 完全一致：本地拆词→预筛→关系分组(6线程)→跨批定向合并→归并审计→分批补全
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

API_URL = 'https://open.bigmodel.cn/api/paas/v4/chat/completions'
API_KEY = os.environ.get('ZHIPU_API_KEY', '')
MODEL = os.environ.get('ZHIPU_MODEL', 'glm-5.3-flash')
MAXF = float('inf')


class ContentFilterError(Exception):
    pass


def call_once(sys_prompt, max_tokens, temperature=0.2, attempts=4):
    body = {
        'model': MODEL,
        'messages': [{'role': 'system', 'content': sys_prompt},
                     {'role': 'user', 'content': '请分析并返回JSON'}],
        'temperature': temperature,
        'response_format': {'type': 'json_object'},
        'max_tokens': max_tokens,
    }
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(API_URL, data=json.dumps(body).encode('utf-8'),
                                         headers={'Content-Type': 'application/json',
                                                  'Authorization': 'Bearer ' + API_KEY})
            resp = urllib.request.urlopen(req, timeout=300)
            data = json.loads(resp.read().decode('utf-8'))
            choice = data['choices'][0]
            content = choice['message'].get('content') or ''
            if choice.get('finish_reason') == 'length' or not content.strip():
                if i < attempts - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError('截断或空内容')
            cleaned = content.replace('```json', '').replace('```', '').strip()
            try:
                return json.loads(cleaned)
            except Exception:
                a, b = cleaned.find('{'), cleaned.rfind('}')
                if a >= 0 and b > a:
                    return json.loads(cleaned[a:b + 1])
                a, b = cleaned.find('['), cleaned.rfind(']')
                if a >= 0 and b > a:
                    return json.loads(cleaned[a:b + 1])
                raise
        except urllib.error.HTTPError as e:
            code = e.code
            txt = e.read().decode('utf-8', 'replace')[:300]
            last = 'HTTP%d %s' % (code, txt)
            if '1301' in txt or 'contentFilter' in txt or '敏感内容' in txt:
                raise ContentFilterError('内容安全过滤: ' + txt[:120])
            if code in (429, 500, 502, 503) and i < attempts - 1:
                time.sleep(3 * (i + 1))
                continue
            raise RuntimeError(last)
        except RuntimeError:
            raise
        except Exception as e:
            last = str(e)
            if i < attempts - 1:
                time.sleep(3 * (i + 1))
                continue
            raise RuntimeError(last)
    raise RuntimeError(last or 'unknown')


def extract_words(text):
    parts = re.split(r'[\s,，、;；。.！!？?：:；（）()【】\[\]「」『』"\'“”·…—\-|/\\]+', text)
    out = []
    seen = set()
    for p in parts:
        p = re.sub(r'^(?:第?[0-9一二三四五六七八九十]{1,3}[、.．)）:：,，])', '', p.strip())
        if not p or not re.search(r'[\u4e00-\u9fff]', p) or len(p) > 8 or len(p) < 2:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def bigrams(w):
    return {w[i:i + 2] for i in range(len(w) - 1)} or {w}


def infer_type(terms):
    four = sum(1 for t in terms if len(t) >= 4)
    return '成语' if four * 2 >= len(terms) else '实词'


class UF:
    def __init__(self):
        self.p = {}

    def add(self, x):
        self.p.setdefault(x, x)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def groups(self):
        m = {}
        for k in self.p:
            m.setdefault(self.find(k), []).append(k)
        return list(m.values())


def pool_run(n, jobs, on_done):
    done = [0]
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(j) for j in jobs]
        for f in futs:
            f.result()
            done[0] += 1
            on_done(done[0])


def run(words_text, lib):
    print('[阶段0] 本地拆词')
    raw_words = extract_words(words_text)
    print('  拆出 %d 词' % len(raw_words))
    if not raw_words:
        raise RuntimeError('未识别出有效词语')

    # 预筛：库内已有跳过；形近库组+单词语库组进摘要
    lib_term_map = {}
    for g in lib:
        for w in g.get('words', []):
            lib_term_map.setdefault(w.get('term', ''), g)
    fresh, skipped = [], []
    seen = set()
    for w in raw_words:
        if w in lib_term_map:
            skipped.append(w)
        elif w not in seen:
            seen.add(w)
            fresh.append(w)
    print('  跳过库内已有 %d 词，新词 %d 个' % (len(skipped), len(fresh)))
    if not fresh:
        return {'result': [], 'skipped': len(skipped), 'note': '全部词语词库中已存在'}

    fresh_bg = set()
    for w in fresh:
        fresh_bg |= bigrams(w)
    scored = []
    for g in lib:
        terms = [w.get('term', '') for w in g.get('words', [])]
        hit = sum(len(bigrams(t) & fresh_bg) for t in terms)
        if hit:
            scored.append((hit, g, terms))
    scored.sort(key=lambda x: -x[0])
    digest = [{'id': g['id'], 'terms': terms} for _, g, terms in scored[:120]]
    digest_ids = {d['id'] for d in digest}
    for g in lib:
        terms = [w.get('term', '') for w in g.get('words', [])]
        if len(terms) == 1 and g['id'] not in digest_ids:
            digest.append({'id': g['id'], 'terms': terms})
            digest_ids.add(g['id'])
    digest_tag = {('L%d' % (i + 1)): d for i, d in enumerate(digest)}

    cross_batch = len(fresh) <= 150
    cross_line = ('本次任务全部词语（供跨批配对参考）：\n' + '、'.join(fresh)) if cross_batch \
        else '（词数较多：本批只做批内配对，与其他批次词的合并由随后的定向合并阶段统一处理）'

    # ===== 阶段1：关系分组 =====
    chunks = [fresh[i:i + 40] for i in range(0, len(fresh), 40)]
    chunk_results = {}

    filtered_words = set()

    def group_one(ci):
        chunk = chunks[ci]
        try:
            res = call_once(build_group_prompt(chunk), 16384)
            record_group(ci, res)
            return
        except ContentFilterError:
            print('  第%d批触发内容安全过滤，二分隔离敏感词…' % (ci + 1))

        def bisect(seg):
            if len(seg) == 1:
                filtered_words.add(seg[0])
                print('    已隔离敏感词:', seg[0])
                return
            mid = len(seg) // 2
            try:
                res = call_once(build_group_prompt(seg[:mid]), 16384)
                record_group(ci, res)
            except ContentFilterError:
                bisect(seg[:mid])
            try:
                res = call_once(build_group_prompt(seg[mid:]), 16384)
                record_group(ci, res)
            except ContentFilterError:
                bisect(seg[mid:])

        bisect(chunk)

    def build_group_prompt(chunk):
        rule5 = ('5. 「本次任务其他批次的词」里若有与本批词易混的，也可以直接组成 pair（跨批配对，词面原样复制）。'
                 if cross_batch else
                 '5. 只对本批词语配对；与其他批次词语的易混关系交由随后的定向合并阶段统一处理。')
        digest_lines = '\n'.join('L%d：%s' % (i + 1, '/'.join(d['terms'])) for i, d in enumerate(digest)) or '（词库暂无可归并的组）'
        lib_section = '词库已有组（新词与之易混时填入 lib）：\n' + digest_lines
        sys_p = '''你是公考行测言语理解「逻辑填空」易混词辨析专家，熟悉历年真题高频易混词考点。

任务：从「本批词语」中找出互为易混词的词，输出易混关系对。只找关系，不写释义。

易混判定（公考标准）：
1. 同对的词必须语义相近但侧重点/感情色彩/搭配对象不同，或形近音近，且是公考逻辑填空常考的对比选项（如：不谋而合/不约而同；耳提面命/谆谆教诲；南辕北辙/背道而驰——通常可出现在同一道逻辑填空的选项中）。
2. 语义无关的词不要出现在任何 pair 里（它们会自动单独成组）。拿不准的不要输出，宁可少配。
3. 每对 2-5 个词，词必须原样从词表中复制。
4. 「词库已有组」中若有与本批词语构成易混关系的组，把该词写进 lib（值填对应 L 标记），不要写进 pairs——新词会归入词库的那一组。
''' + rule5 + '''

输出严格JSON（不要解释）：
{"pairs":[["不谋而合","不约而同"],["南辕北辙","背道而驰"]],"lib":{"望其项背":"L3"}}
没有可配对的就输出 {"pairs":[],"lib":{}}。

''' + lib_section + '''

''' + cross_line + '''

本批词语（重点分析这些）：
''' + '、'.join(chunk)
        return sys_p

    print('[阶段1/3] 关系分组：%d 批（6并发）' % len(chunks))
    def record_group(ci, res):
        cur = chunk_results.setdefault(ci, {'pairs': [], 'lib': {}})
        if isinstance(res, dict):
            cur['pairs'].extend(res.get('pairs', []))
            lib = res.get('lib', {})
            if isinstance(lib, dict):
                cur['lib'].update(lib)

    prog = {'done': 0}
    pool_run(6, [(lambda ci=ci: group_one(ci)) for ci in range(len(chunks))],
             lambda d: print('  分组 %d/%d' % (d, len(chunks))))

    # 聚合
    uf = UF()
    for w in fresh:
        uf.add(w)
    fresh_set = set(fresh)
    lib_merges = {}
    for ci in sorted(chunk_results):
        res = chunk_results[ci]
        for pair in res['pairs']:
            ws = [t for t in pair if t in fresh_set]
            for i in range(1, len(ws)):
                uf.union(ws[0], ws[i])
        for w, tag in res['lib'].items():
            d = digest_tag.get(str(tag))
            if d and w in fresh_set and w not in d['terms']:
                d['terms'].append(w)
                lib_merges.setdefault(d['id'], []).append(w)
    groups = [{'type': infer_type(g), 'terms': g} for g in uf.groups() if g]
    print('  聚合为 %d 组' % len(groups))

    # ===== 跨批定向合并 =====
    if len(fresh) > 150:
        slices = [groups[i:i + 200] for i in range(0, len(groups), 200)]
        cu = UF()
        for i in range(len(groups)):
            cu.add('g%d' % i)
        for si, seg in enumerate(slices):
            print('[阶段1/3] 跨批定向合并：第 %d/%d 轮' % (si + 1, len(slices)))
            lines = '\n'.join('G%d：%s' % (k + 1, '/'.join(g['terms'])) for k, g in enumerate(seg))
            sys_p = '''你是公考行测言语理解「逻辑填空」易混词辨析专家。下面 %d 个小组是分批自动分组产生的，批次之间互相看不见，可能存在：同一考点的词被拆进了不同小组。

任务：找出应当合并的小组对。规则：
1. 只在「确实属于同一考点、做题时可以互相作为混淆选项」的小组之间输出合并关系。
2. 每项 ["G3","G8"] 表示 G3 和 G8 应合并为一个考点组；三个组同考点可写 ["G3","G8","G15"]。
3. 拿不准的不要输出；语义无关的绝不硬配；宁可少输出也不要错配。
4. 组标记必须原样引用（如 G3）。

输出严格JSON（不要解释）：
{"merges":[["G1","G7"]]}
没有需要合并的输出 {"merges":[]}。

小组清单：
''' % len(seg) + lines
            try:
                res = call_once(sys_p, 4096)
            except ContentFilterError:
                print('  定向合并第%d轮触发内容安全过滤，跳过该轮' % (si + 1))
                continue
            def idx_of(tag):
                m = re.match(r'^G(\d+)$', str(tag or '').strip())
                if not m:
                    return -1
                k = int(m.group(1))
                return si * 200 + k - 1 if 1 <= k <= len(seg) else -1
            for pair in (res.get('merges', []) if isinstance(res, dict) else []):
                idxs = [idx_of(t) for t in pair if isinstance(t, str)]
                idxs = [i for i in idxs if 0 <= i < len(groups)]
                for i in range(1, len(idxs)):
                    cu.union('g%d' % idxs[0], 'g%d' % idxs[i])
        merged = []
        for nodes in cu.groups():
            terms = []
            for n in nodes:
                for t in groups[int(n[1:])]['terms']:
                    if t not in terms:
                        terms.append(t)
            merged.append({'type': infer_type(terms), 'terms': terms})
        groups = merged
        print('  定向合并后 %d 组' % len(groups))

    # ===== 归并审计 =====
    multi = [g for g in groups if len(g['terms']) >= 3]
    small_idx = [i for i, g in enumerate(groups) if len(g['terms']) <= 2]
    if small_idx:
        multi_line = '\n'.join('/'.join(g['terms']) for g in multi) or '（无）'
        lib_all = '　'.join('/'.join([w.get('term', '') for w in g.get('words', [])]) for g in lib) or '（词库为空）'
        lib_id_by_terms = {'§'.join([w.get('term', '') for w in g.get('words', [])]): g['id'] for g in lib}
        slices = [small_idx[i:i + 10] for i in range(0, len(small_idx), 10)]
        audit_results = {}

        def audit_one(si):
            seg = slices[si]
            seg_lines = '\n'.join('A%d：%s（%s）' % (k + 1, '/'.join(groups[gi]['terms']),
                                                    '单词语' if len(groups[gi]['terms']) == 1 else groups[gi]['type'])
                                  for k, gi in enumerate(seg))
            sys_p = '''你是公考行测言语理解「逻辑填空」易混词辨析专家。下面这些小组是自动分组产生的，可能存在：本该同属一个考点的词被拆散在不同组、单词语其实与某组易混、词被放错组。

任务：找出应该合并/纠正的关系，输出关系对。只输出关系，不重组整组。
1. merges：每项 [词A,词B]，表示这两个词应属同一考点组（词A、词B来自不同小组，或之一来自已确认组）。2-5个词一组时可用多项表达，如 ["词A","词B","词C"]。
2. lib：若某词其实应并入「词库已有组」，写 {"词":"所在库组的完整词组"}，值必须从「词库全部组」里原样复制一个完整词组。
3. 语义无关的不要强行输出。宁可少输出，不要错配。
4. 只处理「待审小组」里的词，词面原样复制。

输出严格JSON（不要解释）：
{"merges":[["词A","词B"]],"lib":{"某词":"某词/某词2/某词3"}}
没有要调整的输出 {"merges":[],"lib":{}}。

已确认组（3词以上，默认正确，仅供对照）：
''' + multi_line + '''

待审小组：
''' + seg_lines + '''

词库全部组（lib 的值必须从这里原样复制）：
''' + lib_all
            try:
                res = call_once(sys_p, 8192)
                audit_results[si] = {'merges': res.get('merges', []) if isinstance(res, dict) else [],
                                     'lib': res.get('lib', {}) if isinstance(res, dict) else {}}
            except ContentFilterError:
                print('  审计片%d触发内容安全过滤，跳过该片（组保持原样）' % (si + 1))
                audit_results[si] = {'merges': [], 'lib': {}}

        print('[阶段2/3] 归并审计：%d 片（6并发）' % len(slices))
        pool_run(6, [(lambda si=si: audit_one(si)) for si in range(len(slices))],
                 lambda d: print('  审计 %d/%d' % (d, len(slices))))

        au = UF()
        for i in range(len(groups)):
            au.add('g%d' % i)
        lib_merges2 = {}
        for si in sorted(audit_results):
            res = audit_results[si]
            for pair in res['merges']:
                def idx_of_word(w, _pair=pair):
                    return next((i for i, g in enumerate(groups) if w in g['terms']), -1)
                idxs = [idx_of_word(w) for w in pair if isinstance(w, str)]
                idxs = [i for i in idxs if i >= 0]
                for i in range(1, len(idxs)):
                    au.union('g%d' % idxs[0], 'g%d' % idxs[i])
            for w, grp in res['lib'].items():
                key = '§'.join(x.strip() for x in str(grp).split('/'))
                gid = lib_id_by_terms.get(key)
                if gid and any(w in g['terms'] for g in groups):
                    lib_merges2.setdefault(gid, []).append(w)
        merged = []
        for nodes in au.groups():
            terms = []
            for n in nodes:
                for t in groups[int(n[1:])]['terms']:
                    if t not in terms:
                        terms.append(t)
            merged.append({'type': infer_type(terms), 'terms': terms})
        final_groups = merged
        all_lib = dict(lib_merges)
        for gid, ws in lib_merges2.items():
            all_lib.setdefault(gid, [])
            all_lib[gid].extend([w for w in ws if w not in all_lib[gid]])
    else:
        final_groups = groups
        all_lib = lib_merges

    print('  审计后 %d 组' % len(final_groups))

    # ===== 补全 =====
    entries = [{'type': g['type'], 'terms': g['terms'], 'libTargetId': None} for g in final_groups if g['terms']]
    for gid, ws in all_lib.items():
        g = next((x for x in lib if x['id'] == gid), None)
        g_terms = [w.get('term', '') for w in g.get('words', [])] if g else []
        room = len(g.get('words', [])) if g else 0
        terms = [w for w in ws if w not in g_terms][:max(0, 5 - room) if g else len(ws)]
        if terms:
            entries.append({'type': g['type'] if g else '实词', 'terms': terms, 'libTargetId': gid})
    batches = []
    cur, curw = [], 0
    for e in entries:
        if curw and curw + len(e['terms']) > 15:
            batches.append(cur)
            cur, curw = [], 0
        cur.append(e)
        curw += len(e['terms'])
    if cur:
        batches.append(cur)

    def fill_one(bi):
        batch = batches[bi]
        listing = '\n'.join('组%d（%s）：%s' % (i + 1, e['type'], ' / '.join(e['terms'])) for i, e in enumerate(batch))
        wc = sum(len(e['terms']) for e in batch)
        sys_p = '''你是公考行测言语理解「逻辑填空」词语辨析专家。为下列每组词语按公考考点补全内容。

要求：
1. term：原样返回输入词语，不得改写；以完整的词/成语为单位分析，绝不拆成单个汉字逐字解释。
2. meaning：简明词典义，25字内。
3. keyPoint：这个词语的"使用画像"——一整句连贯白话，70字内，让考生只看这一条就能决定逻辑填空该不该选它。内部从五个维度分析：词义侧重、感情色彩、搭配对象、语义轻重、语法功能；只写与同组其他词有差异的维度，差异必须具体到能直接做题（具体搭配对象、明确的褒贬、语法限制、经典易错本义）。维度融在句子里，不加标签、不写"与某词相比"、不写"侧重点不同"这类空洞话。单词语组则写该词的易错考点。
4. example：用该词造一个逻辑填空风格例句，35字内，用法必须正确。
5. 保持分组不变，每个完整的词/成语各输出一条（不要按字拆分），只输出JSON（不要解释）：
{"groups":[{"t":"成语","words":[{"term":"","meaning":"","keyPoint":"","example":""}]}]}

keyPoint 合格示范（照这个水准写）：
推脱 → 只用于推卸自己该负的责任和罪责，含贬义，不能用来拒绝邀请。
推托 → 指借故婉拒别人的请求或邀请，如推托有事，不含推卸责任之义。
差强人意 → 指大体上还算令人满意（差=稍微），是褒义词，误解为不满意是高频错误。
首当其冲 → 指最先遭受攻击或灾难，是被动受害，绝不能理解为首先带头。
望其项背 → 指能赶得上，多用于否定式，肯定式用法是错的。
屡试不爽 → 指屡次试验都没有差错（爽=违背），不是屡次失败，别望文生义。
空穴来风 → 本指消息有根据并非无中生有，做题时优先按有根据理解。
耳提面命 → 形容长辈恳切地教导，只用于长辈对晚辈，平辈之间不能用。
蔚然成风 → 指好的风气逐渐形成，含褒义，只能形容好现象，不能形容坏风气。

待填写词组：
''' + listing
        try:
            res = call_once(sys_p, max(4096, min(8192, wc * 260 + 2500)), 0.3)
        except ContentFilterError:
            print('  补全批%d触发内容安全过滤，该批内容留空（词仍导入）' % (bi + 1))
            fill_results[bi] = []
            return
        glist = res.get('groups', []) if isinstance(res, dict) else []
        fill_results[bi] = glist

    print('[阶段3/3] 补全：%d 批（6并发）' % len(batches))
    fill_results = {}
    pool_run(6, [(lambda bi=bi: fill_one(bi)) for bi in range(len(batches))],
             lambda d: print('  补全 %d/%d' % (d, len(batches))))

    # 组装（与前端 assembleImportList 一致）
    result = []
    for bi, batch in enumerate(batches):
        glist = fill_results.get(bi, [])
        for ei, e in enumerate(batch):
            g = glist[ei] if ei < len(glist) else None
            by_term = {str(w.get('term', '')).replace(' ', ''): w for w in (g.get('words', []) if isinstance(g, dict) else [])}
            words = []
            for t in e['terms']:
                hit = by_term.get(t, {})
                words.append({'term': t,
                              'meaning': str(hit.get('meaning', '')).strip(),
                              'keyPoint': str(hit.get('keyPoint', '')).strip(),
                              'example': str(hit.get('example', '')).strip()})
            result.append({'type': e['type'], 'words': words, 'matchTargetGroupId': e['libTargetId']})

    print('完成：%d 组，跳过 %d 词' % (len(result), len(skipped)))
    return {'result': result, 'skipped': len(skipped), 'filtered': sorted(filtered_words)}


if __name__ == '__main__':
    if not API_KEY:
        sys.exit('ZHIPU_API_KEY not set')
    words_text = open('cloud/words.txt', encoding='utf-8').read()
    lib = json.load(open('cloud/lib.json', encoding='utf-8'))
    out = run(words_text, lib)
    json.dump(out, open('cloud/result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('result.json written')
