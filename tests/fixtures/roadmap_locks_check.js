// src/static/index.html의 로드맵 잠금(locked) 로직을 검증한다.
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

// 2026-08-05 9단계 재구성 - 사전검토ᆞ설계서류ᆞ신청접수 substep은 이제
// (옛 step1Substeps 하나 대신) preReviewSubsteps/designDocsSubsteps/
// applicationSubsteps 3개로 나뉘어 steps 배열 리터럴 안에서 바로 참조된다.
// 이 테스트는 그 항목들의 실제 판정 로직(diag/docs 파싱)이 아니라 다른
// Step(공사ᆞ사용승인ᆞ창업행정ᆞ오픈준비ᆞ건축사배지)의 잠금 로직을 검증하는
// 목적이라, 예전 step1Substeps처럼 빈 배열을 그대로 주입해 대체한다.
function buildSteps({
  latestFoodResult, latestFireResult, latestSignageResult, taskProgress,
  actType, latestPermitResult, caseTypeResult, requiresArchitect = null, permitNotNeeded = false,
  factInfoLines = [], preReviewSubsteps = [], designDocsSubsteps = [], applicationSubsteps = [],
  applyStepTitle = "허가ᆞ신고 신청ᆞ접수",
  // 2026-08-05: Step1 체크리스트가 lastAddress/caseFacts를 직접 참조하도록
  // 바뀌면서 새로 필요해진 자유변수 - 이 테스트는 Step1 자체를 검증하지
  // 않으므로 기본값만 준다.
  lastAddress = null, caseFacts = {},
}) {
  // renderRoadmap()의 classified(buildChecklists()와 동일 정의)를 그대로
  // 재현 - "신고ᆞ허가 대상 판단" Step의 substep이 참조한다.
  const classified = !!(latestPermitResult && latestPermitResult.permit_type);
  const fn = new Function(
    "latestFoodResult", "latestFireResult", "latestSignageResult", "taskProgress",
    "actType", "latestPermitResult", "caseTypeResult", "requiresArchitect", "permitNotNeeded", "classified",
    "factInfoLines", "preReviewSubsteps", "designDocsSubsteps", "applicationSubsteps", "applyStepTitle",
    "lastAddress", "caseFacts",
    block + "\nreturn steps;"
  );
  return fn(
    latestFoodResult, latestFireResult, latestSignageResult, taskProgress,
    actType, latestPermitResult, caseTypeResult, requiresArchitect, permitNotNeeded, classified,
    factInfoLines, preReviewSubsteps, designDocsSubsteps, applicationSubsteps, applyStepTitle,
    lastAddress, caseFacts
  );
}

// renderRoadmap()의 실제 분기(sub.field가 있으면 locked&&!done일 때만 잠금
// 표시, 없으면 locked만으로 잠금 표시)를 그대로 흉내낸다.
function isVisuallyLocked(sub) {
  if (sub.info) return false;
  if (sub.notApplicable) return false;  // 해당없음(✕)이 잠금보다 우선 - 실제 렌더와 동일
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

const commonArgs = { actType: "신축", latestPermitResult: null, caseTypeResult: null };
const results = [];
function check(label, passed) {
  results.push({ label, passed: !!passed });
}

// 식품위생 영업신고 - 사용승인 전/후 (2026-08-05: "Step 3 · 창업 행정" →
// "사업자등록 및 위생교육"으로 재배치)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: { use_approval: false } });
  check("food: 사용승인 전엔 판정 나도 잠김", isVisuallyLocked(findSub(steps, "사업자등록 및 위생교육", "식품위생")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: { use_approval: true } });
  check("food: 사용승인 후엔 잠금 해제", !isVisuallyLocked(findSub(steps, "사업자등록 및 위생교육", "식품위생")));
}

// 소방시설 - 공사(시공) 전/후 (2026-08-05: "Step 2 · 공사" →
// "사용승인ᆞ완공검사"로 재배치)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("fire: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "사용승인ᆞ완공검사", "소방시설")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: true } });
  check("fire: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "사용승인ᆞ완공검사", "소방시설")));
}

// 간판 - 공사(시공) 전/후 (2026-08-05: "Step 3 · 창업 행정" →
// "간판신고 및 오픈 준비"로 재배치)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("signage: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "간판")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { construction: true } });
  check("signage: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "간판")));
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
    const locked = isVisuallyLocked(findSub(steps, "사업자등록 및 위생교육", "사업자등록"));
    check(`business_registration: documents_prepared=${docs}, food=${food} → locked=${expectLocked}`, locked === expectLocked);
  }

  // 인허가불필요 케이스는 documents_prepared가 영원히 안 채워진다 -
  // permitNotNeeded=true면 documents_prepared 없이도 잠금이 풀려야 한다
  // (2026-08-04, 사용승인까지 끝났는데 Lock이 안 풀린다는 신고).
  const steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: { documents_prepared: false }, permitNotNeeded: true });
  check("business_registration: permitNotNeeded=true면 documents_prepared 없어도 잠금 해제", !isVisuallyLocked(findSub(steps, "사업자등록 및 위생교육", "사업자등록")));
}

// 직원 등록 - 사업자등록 여부 + hires_staff 예외 (2026-08-05: "Step 4 · 오픈
// 준비" → "간판신고 및 오픈 준비"로 재배치)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { business_registration: false } });
  check("staff: 사업자등록 전엔 잠김", isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "직원 등록")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { business_registration: true } });
  check("staff: 사업자등록 후엔 해제", !isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "직원 등록")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { business_registration: false, hires_staff: false } });
  check("staff: hires_staff=false면 사업자등록 전이어도 안 잠김(done 우회)", !isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "직원 등록")));
}

// 영업 시작 - 식품위생ᆞ간판ᆞ사업자등록ᆞ위생교육 전부 완료해야 해제
{
  let steps = buildSteps({
    ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: "신고불필요",
    taskProgress: { business_registration: true, hygiene_education: false },
  });
  check("opened: 4개 중 1개(위생교육)만 빠져도 잠김", isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "영업 시작")));

  steps = buildSteps({
    ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: "신고불필요",
    taskProgress: { business_registration: true, hygiene_education: true },
  });
  check("opened: 4개 전부 충족되면 해제", !isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "영업 시작")));
}

// 건축사 대행 배지 - requiresArchitect 값이 architect 필드로 그대로
// 반영되는지. 배지는 true(건축사 필수)일 때만 렌더하고 false(개인 직접
// 가능)ᆞnull은 배지를 표시하지 않는다(false는 채팅 안내로 대체). 2026-08-05
// 9단계 재구성으로 "Step 0 · 건축 유형 + 대상 판정"에서 실제로 건축사를
// 선정하는 "건축사사무소 선정 및 도면 작성"으로 배지 위치도 옮겼다.
{
  const designStep = (ra) => buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: {}, requiresArchitect: ra }).find((s) => s.title === "건축사사무소 선정 및 도면 작성");
  check("architect: true면 architect=true(대행 배지 표시)", designStep(true).architect === true);
  check("architect: false면 배지 미표시(필드만 false)", designStep(false).architect === false);
  check("architect: null이면 배지 없음", designStep(null).architect === null || designStep(null).architect === undefined);
}

// 해당없음(✕) - 대화로 필요 없다고 확인된 항목은 notApplicable로 닫힌다.
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: {} });
  check("na: 소방시설 대상 없음(빈 배열)이면 해당없음", !!findSub(steps, "사용승인ᆞ완공검사", "소방시설").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "허가ᆞ신고 불필요", taskProgress: {} });
  check("na: 간판 신고 불필요면 해당없음", !!findSub(steps, "간판신고 및 오픈 준비", "간판").notApplicable);
  check("na: 소방시설 있으면 해당없음 아님", !findSub(steps, "사용승인ᆞ완공검사", "소방시설").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: "신고대상제외", latestFireResult: ["소화기구"], latestSignageResult: null, taskProgress: {} });
  check("na: 식품위생 신고대상제외면 해당없음", !!findSub(steps, "사업자등록 및 위생교육", "식품위생").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { hires_staff: false } });
  check("na: 직원 없음이면 직원 등록 해당없음", !!findSub(steps, "간판신고 및 오픈 준비", "직원 등록").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: "신고대상제외", latestFireResult: [], latestSignageResult: null, taskProgress: {} });
  check("na: 식품위생 신고대상제외면 위생교육도 해당없음", !!findSub(steps, "사업자등록 및 위생교육", "위생교육").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: {} });
  check("na: 식품위생 대상이면 위생교육은 해당없음 아님", !findSub(steps, "사업자등록 및 위생교육", "위생교육").notApplicable);
}

// 인테리어 전용 케이스 - 착공신고ᆞ시공이 notApplicable로 닫히는지(2026-08-04)
// (2026-08-05: 착공신고ᆞ시공은 "착공신고 및 공사"에, 소방시설은
// "사용승인ᆞ완공검사"에, 간판은 "간판신고 및 오픈 준비"에 재배치)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { interior_only: true } });
  // "착공신고"만 접두어로 쓰면 2026-08-05에 추가된 참고용 안내 줄
  // ("착공신고서ᆞ설계도서 제출...")도 같은 접두어라 find()가 그쪽을 먼저
  // 집을 수 있다 - 실제 토글 대상 행("착공신고 완료 확인")을 명시적으로 지정.
  check("interior_only: 착공신고 해당없음", !!findSub(steps, "착공신고 및 공사", "착공신고 완료 확인").notApplicable);
  check("interior_only: 시공 해당없음", !!findSub(steps, "착공신고 및 공사", "시공 완료 확인").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: null, taskProgress: { interior_only: true } });
  check("interior_only: 소방시설은 construction 없이도 잠금 해제", !isVisuallyLocked(findSub(steps, "사용승인ᆞ완공검사", "소방시설")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { interior_only: true } });
  check("interior_only: 간판도 construction 없이도 잠금 해제", !isVisuallyLocked(findSub(steps, "간판신고 및 오픈 준비", "간판")));
}

process.stdout.write(JSON.stringify(results));
