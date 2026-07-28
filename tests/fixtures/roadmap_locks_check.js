// src/static/index.html의 로드맵 Step3ᆞ4 잠금(locked) 로직을 검증한다.
// 코드를 다시 옮겨 적지 않고 파일에서 실제 블록을 그대로 잘라내 실행한다 -
// 그래야 index.html이 바뀌면 이 테스트도 최신 코드를 대상으로 돈다.
//
// 출력은 tests/test_roadmap_locks.py가 파싱할 수 있게 JSON 배열
// [{label, passed}, ...] 하나만 stdout에 찍는다(pytest 쪽에서 subprocess로
// 감싸 각 항목을 개별 assert로 쪼갠다).
const fs = require("fs");
const path = require("path");

const INDEX_HTML = path.join(__dirname, "..", "..", "src", "static", "index.html");
const html = fs.readFileSync(INDEX_HTML, "utf8");

const startMarker = "const step3AllDone = latestFoodResult != null";
const startIdx = html.indexOf(startMarker);
if (startIdx === -1) {
  throw new Error("index.html에서 'const step3AllDone = ...' 블록을 못 찾았습니다 - 코드가 바뀌었으면 이 fixture의 startMarker도 같이 갱신하세요.");
}
const endMarker = "\n    ];\n";
const endIdx = html.indexOf(endMarker, startIdx) + endMarker.length;
const block = html.slice(startIdx, endIdx);

function buildSteps({ latestFoodResult, latestFireResult, latestSignageResult, taskProgress, actType, step1Substeps, latestPermitResult, caseTypeResult }) {
  const fn = new Function(
    "latestFoodResult", "latestFireResult", "latestSignageResult", "taskProgress",
    "actType", "step1Substeps", "latestPermitResult", "caseTypeResult",
    block + "\nreturn steps;"
  );
  return fn(latestFoodResult, latestFireResult, latestSignageResult, taskProgress, actType, step1Substeps, latestPermitResult, caseTypeResult);
}

// renderRoadmap()의 실제 분기(sub.field가 있으면 locked&&!done일 때만 잠금
// 표시, 없으면 locked만으로 잠금 표시)를 그대로 흉내낸다 - index.html:1090,
// 1119 참고.
function isVisuallyLocked(sub) {
  if (sub.info) return false;
  if (sub.field) return !!sub.locked && !sub.done;
  return !!sub.locked;
}

function findSub(steps, stepTitle, textPrefix) {
  const step = steps.find((s) => s.title === stepTitle);
  if (!step) throw new Error(`Step "${stepTitle}"을 못 찾았습니다`);
  const sub = step.substeps.find((s) => s.text === textPrefix || s.text.startsWith(textPrefix));
  if (!sub) throw new Error(`"${stepTitle}" 안에서 "${textPrefix}" 항목을 못 찾았습니다`);
  return sub;
}

const commonArgs = { actType: "신축", step1Substeps: [], latestPermitResult: null, caseTypeResult: null };
const results = [];
function check(label, passed) {
  results.push({ label, passed: !!passed });
}

// 식품위생 영업신고 - 사용승인 전/후
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: { use_approval: false } });
  check("food: 사용승인 전엔 판정 나도 잠김", isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "식품위생")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: { use_approval: true } });
  check("food: 사용승인 후엔 잠금 해제", !isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "식품위생")));
}

// 소방시설ᆞ간판 - 공사(시공) 전/후
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("fire: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "소방시설")));
  check("signage: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "간판")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: true } });
  check("fire: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "소방시설")));
  check("signage: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "간판")));
}

// 사업자등록 - 서류ᆞ영업신고 판정 조합
{
  const cases = [
    [false, null, true],
    [true, null, true],
    [false, "일반음식점", true],
    [true, "일반음식점", false],
  ];
  for (const [docs, food, expectLocked] of cases) {
    const steps = buildSteps({ ...commonArgs, latestFoodResult: food, latestFireResult: [], latestSignageResult: null, taskProgress: { documents_prepared: docs } });
    const locked = isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "사업자등록"));
    check(`business_registration: documents_prepared=${docs}, food=${food} → locked=${expectLocked}`, locked === expectLocked);
  }
}

// 직원 등록 - 사업자등록 여부 + hires_staff 예외
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { business_registration: false } });
  check("staff: 사업자등록 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 4 · 오픈 준비", "직원 등록")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { business_registration: true } });
  check("staff: 사업자등록 후엔 해제", !isVisuallyLocked(findSub(steps, "Step 4 · 오픈 준비", "직원 등록")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { business_registration: false, hires_staff: false } });
  check("staff: hires_staff=false면 사업자등록 전이어도 안 잠김(done 우회)", !isVisuallyLocked(findSub(steps, "Step 4 · 오픈 준비", "직원 등록")));
}

// 영업 시작 - Step3 5개 항목 전부 완료해야 해제
{
  let steps = buildSteps({
    ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: "신고불필요",
    taskProgress: { business_registration: true, hygiene_education: false },
  });
  check("opened: 5개 중 1개(위생교육)만 빠져도 잠김", isVisuallyLocked(findSub(steps, "Step 4 · 오픈 준비", "영업 시작")));

  steps = buildSteps({
    ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: "신고불필요",
    taskProgress: { business_registration: true, hygiene_education: true },
  });
  check("opened: 5개 전부 충족되면 해제", !isVisuallyLocked(findSub(steps, "Step 4 · 오픈 준비", "영업 시작")));
}

process.stdout.write(JSON.stringify(results));
