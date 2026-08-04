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

function buildSteps({ latestFoodResult, latestFireResult, latestSignageResult, taskProgress, actType, step1Substeps, latestPermitResult, caseTypeResult, requiresArchitect = null, permitNotNeeded = false }) {
  // renderRoadmap()의 classified(801행 buildChecklists()와 동일 정의)를 그대로
  // 재현 - Step0의 "대상 판정" substep이 참조한다.
  const classified = !!(latestPermitResult && latestPermitResult.permit_type);
  const fn = new Function(
    "latestFoodResult", "latestFireResult", "latestSignageResult", "taskProgress",
    "actType", "step1Substeps", "latestPermitResult", "caseTypeResult", "requiresArchitect", "permitNotNeeded", "classified",
    block + "\nreturn steps;"
  );
  return fn(latestFoodResult, latestFireResult, latestSignageResult, taskProgress, actType, step1Substeps, latestPermitResult, caseTypeResult, requiresArchitect, permitNotNeeded, classified);
}

// renderRoadmap()의 실제 분기(sub.field가 있으면 locked&&!done일 때만 잠금
// 표시, 없으면 locked만으로 잠금 표시)를 그대로 흉내낸다 - index.html:1090,
// 1119 참고.
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

// 소방시설(Step2) - 공사(시공) 전/후
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("fire: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 2 · 공사", "소방시설")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "신고", taskProgress: { construction: true } });
  check("fire: 공사 후엔 해제", !isVisuallyLocked(findSub(steps, "Step 2 · 공사", "소방시설")));
}

// 간판(Step3) - 공사(시공) 전/후
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { construction: false } });
  check("signage: 공사 전엔 잠김", isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "간판")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { construction: true } });
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

  // 인허가불필요 케이스는 documents_prepared가 영원히 안 채워진다 -
  // permitNotNeeded=true면 documents_prepared 없이도 잠금이 풀려야 한다
  // (2026-08-04, 사용승인까지 끝났는데 Lock이 안 풀린다는 신고).
  const steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: { documents_prepared: false }, permitNotNeeded: true });
  check("business_registration: permitNotNeeded=true면 documents_prepared 없어도 잠금 해제", !isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "사업자등록")));
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

// 건축사 대행 배지(Step 0) - requiresArchitect 값이 Step 0의 architect 필드로
// 그대로 반영되는지. 배지는 true(건축사 필수)일 때만 렌더하고 false(개인 직접
// 가능)ᆞnull은 배지를 표시하지 않는다(false는 채팅 안내로 대체). 판정 축이
// Step0로 옮겨오면서(2026-08-04) 배지도 Step1에서 Step0로 함께 이동했다.
{
  const step1 = (ra) => buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: {}, requiresArchitect: ra }).find((s) => s.title === "Step 0 · 건축 유형 + 대상 판정");
  check("architect: true면 Step0 architect=true(대행 배지 표시)", step1(true).architect === true);
  check("architect: false면 배지 미표시(필드만 false)", step1(false).architect === false);
  check("architect: null이면 배지 없음", step1(null).architect === null || step1(null).architect === undefined);
}

// 해당없음(✕) - 대화로 필요 없다고 확인된 항목은 notApplicable로 닫힌다.
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: {} });
  check("na: 소방시설 대상 없음(빈 배열)이면 해당없음", !!findSub(steps, "Step 2 · 공사", "소방시설").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: "허가ᆞ신고 불필요", taskProgress: {} });
  check("na: 간판 신고 불필요면 해당없음", !!findSub(steps, "Step 3 · 창업 행정", "간판").notApplicable);
  check("na: 소방시설 있으면 해당없음 아님", !findSub(steps, "Step 2 · 공사", "소방시설").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: "신고대상제외", latestFireResult: ["소화기구"], latestSignageResult: null, taskProgress: {} });
  check("na: 식품위생 신고대상제외면 해당없음", !!findSub(steps, "Step 3 · 창업 행정", "식품위생").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { hires_staff: false } });
  check("na: 직원 없음이면 직원 등록 해당없음", !!findSub(steps, "Step 4 · 오픈 준비", "직원 등록").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: "신고대상제외", latestFireResult: [], latestSignageResult: null, taskProgress: {} });
  check("na: 식품위생 신고대상제외면 위생교육도 해당없음", !!findSub(steps, "Step 3 · 창업 행정", "위생교육").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: "일반음식점", latestFireResult: [], latestSignageResult: null, taskProgress: {} });
  check("na: 식품위생 대상이면 위생교육은 해당없음 아님", !findSub(steps, "Step 3 · 창업 행정", "위생교육").notApplicable);
}

// 인테리어 전용 케이스 - 착공신고ᆞ시공이 notApplicable로 닫히는지(2026-08-04)
{
  let steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: null, taskProgress: { interior_only: true } });
  check("interior_only: 착공신고 해당없음", !!findSub(steps, "Step 2 · 공사", "착공신고").notApplicable);
  check("interior_only: 시공 해당없음", !!findSub(steps, "Step 2 · 공사", "시공").notApplicable);

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: ["소화기구"], latestSignageResult: null, taskProgress: { interior_only: true } });
  check("interior_only: 소방시설은 construction 없이도 잠금 해제", !isVisuallyLocked(findSub(steps, "Step 2 · 공사", "소방시설")));

  steps = buildSteps({ ...commonArgs, latestFoodResult: null, latestFireResult: [], latestSignageResult: "신고", taskProgress: { interior_only: true } });
  check("interior_only: 간판도 construction 없이도 잠금 해제", !isVisuallyLocked(findSub(steps, "Step 3 · 창업 행정", "간판")));
}

process.stdout.write(JSON.stringify(results));
