// src/static/index.html의 renderMarkdown() 번호 목록 렌더링을 검증한다.
// 코드를 다시 옮겨 적지 않고 파일에서 실제 블록을 그대로 잘라내 실행한다 -
// 그래야 index.html이 바뀌면 이 테스트도 최신 코드를 대상으로 돈다.
//
// 출력은 tests/test_markdown_render.py가 파싱할 수 있게 JSON 배열
// [{label, passed}, ...] 하나만 stdout에 찍는다.
const fs = require("fs");
const path = require("path");

const INDEX_HTML = path.join(__dirname, "..", "..", "src", "static", "index.html");
const html = fs.readFileSync(INDEX_HTML, "utf8");

const startMarker = "function escapeHtml(str) {";
const startIdx = html.indexOf(startMarker);
if (startIdx === -1) {
  throw new Error("index.html에서 'function escapeHtml' 블록을 못 찾았습니다 - 코드가 바뀌었으면 이 fixture의 startMarker도 같이 갱신하세요.");
}
const endMarker = "\n  }\n\n  function addMessage";
const endIdx = html.indexOf(endMarker, startIdx);
if (endIdx === -1) {
  throw new Error("index.html에서 renderMarkdown 블록의 끝(endMarker)을 못 찾았습니다.");
}
const block = html.slice(startIdx, endIdx + 4); // "\n  }" 까지 포함

const fn = new Function("text", block + "\nreturn renderMarkdown(text);");

const results = [];
function check(label, passed) {
  results.push({ label, passed: !!passed });
}

// 회귀 테스트(2026-08-04, 세션 3d2f20a1) - 번호 목록 중간에 하위 불릿(-)이
// 끼면 <ol>이 한 번 닫혔다 새로 열리면서, value 속성이 없으면 브라우저가
// 다시 1부터 세어 "1. 2. 1."처럼 보이는 버그가 있었다.
{
  const text = "1. 건물 주소\n2. 건물 상태:\n   - 신축\n   - 대수선\n3. 영업 방식";
  const out = fn(text);
  const values = [...out.matchAll(/<li value="(\d+)">/g)].map((m) => m[1]);
  check("번호 목록이 하위 불릿으로 끊겨도 원문 번호(1,2,3) 그대로 유지", JSON.stringify(values) === JSON.stringify(["1", "2", "3"]));
}

// 중간에 끊김이 없는 평범한 번호 목록도 그대로 잘 나와야 한다.
{
  const text = "1. 첫째\n2. 둘째\n3. 셋째";
  const out = fn(text);
  const values = [...out.matchAll(/<li value="(\d+)">/g)].map((m) => m[1]);
  check("끊김 없는 번호 목록도 정상 렌더", JSON.stringify(values) === JSON.stringify(["1", "2", "3"]));
}

// 빈 줄로 끊긴 뒤 다시 이어지는 번호 목록도 원문 번호를 유지해야 한다.
{
  const text = "1. 첫째\n2. 둘째\n\n3. 셋째";
  const out = fn(text);
  const values = [...out.matchAll(/<li value="(\d+)">/g)].map((m) => m[1]);
  check("빈 줄로 끊긴 번호 목록도 원문 번호 유지", JSON.stringify(values) === JSON.stringify(["1", "2", "3"]));
}

process.stdout.write(JSON.stringify(results));
