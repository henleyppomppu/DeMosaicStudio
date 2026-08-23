using System.Collections.Frozen;
using DeMosaicStudio.Domain.Diagnostics;

namespace DeMosaicStudio.Application.Diagnostics;

/// <summary>
/// What each numbered code means to the person looking at the window, in Korean.
/// <para>
/// Separate from <see cref="ErrorCodes"/> on purpose. The meanings there are the protocol's own,
/// locked to <c>worker/demosaic_worker/errors.py</c> by <c>fixtures/parity/error_codes.json</c>
/// (§13.4) and printed in <c>docs/ERROR_CODES.md</c> — translating them in place would break the
/// parity fixture and change what two implementations agree on. This is display text sitting beside
/// them, and the code itself is always shown too, so a search still lands on the documentation.
/// </para>
/// <para>
/// In Application rather than in the window because completeness is a rule worth a test, and the
/// window cannot be reached from one (AGENTS.md). A code with no line here is a code the user would
/// meet in English.
/// </para>
/// </summary>
public static class ErrorText
{
    private static readonly FrozenDictionary<string, string> Korean =
        new Dictionary<string, string>(StringComparer.Ordinal)
        {
            // E1xxx — 입력 파일
            ["E1001"] = "파일을 찾을 수 없거나 읽을 수 없습니다",
            ["E1002"] = "지원하지 않는 컨테이너 형식입니다",
            ["E1003"] = "지원하지 않는 영상 코덱이거나 프로파일입니다",
            ["E1004"] = "원본이 손상되었습니다: 컨테이너를 풀 수 없습니다",
            ["E1005"] = "원본에 영상 스트림이 없습니다",
            ["E1006"] = "원본의 메타데이터가 서로 맞지 않습니다",

            // E2xxx — 디코딩
            ["E2001"] = "하드웨어 디코더를 초기화하지 못했습니다",
            ["E2002"] = "디코딩 중 오류: 이 프레임은 복구할 수 없습니다",
            ["E2003"] = "디코딩 중 오류: 스트림 전체를 복구할 수 없습니다",
            ["E2004"] = "타임스탬프가 허용 범위를 넘어 끊겼습니다",

            // E3xxx — 탐지·추적
            ["E3001"] = "탐지 모델을 불러오지 못했습니다",
            ["E3002"] = "탐지 추론에 실패했습니다",
            ["E3003"] = "탐지 모델의 출력 형태가 맞지 않습니다",
            ["E3201"] = "트랙 상태 전이 규칙을 위반했습니다",

            // E4xxx — 복원
            ["E4001"] = "복원 모델을 불러오지 못했습니다",
            ["E4002"] = "복원 추론에 실패했습니다",
            ["E4003"] = "이웃 프레임을 하나도 정렬하지 못했습니다",
            ["E4004"] = "영역이 모델의 최소 크기보다 작습니다",
            ["E4401"] = "GPU 메모리가 부족합니다. 완화 단계를 모두 시도했습니다",
            ["E4402"] = "이 모델을 실행할 수 있는 백엔드가 없습니다",

            // E5xxx — 인코딩·먹싱
            ["E5001"] = "인코더를 초기화하지 못했습니다",
            ["E5002"] = "인코딩 중 오류가 났습니다",
            ["E5003"] = "컨테이너로 묶는 데 실패했습니다",
            ["E5004"] = "출력 컨테이너가 원본의 스트림 하나를 담을 수 없습니다",

            // E6xxx — 시스템 자원
            ["E6001"] = "디스크 공간이 부족합니다",
            ["E6002"] = "출력 경로에 쓸 수 없습니다",
            ["E6003"] = "출력 파일을 다른 프로그램이 잠그고 있습니다",
            ["E6004"] = "시스템 메모리가 부족합니다",
            ["E6005"] = "필요한 지원 라이브러리가 없거나 불러올 수 없습니다",

            // E7xxx — 워커·프로토콜
            ["E7001"] = "워커와 프로토콜 주 버전이 다릅니다",
            ["E7002"] = "워커가 초기 교신에 응답하지 않았습니다",
            ["E7003"] = "워커가 이미 다른 작업을 처리하고 있습니다",
            ["E7004"] = "워커가 취소 유예 시간 안에 끝내지 못했습니다",
            ["E7005"] = "워커가 비정상 종료했습니다",
            ["E7006"] = "프로토콜 메시지 형식이 잘못되었습니다",

            // E9xxx
            ["E9001"] = "예상하지 못한 내부 오류입니다",

            // Wxxxx — 경고. 작업을 실패시키지 않습니다 (§10.1)
            ["W1101"] = "소프트웨어 디코딩으로 전환했습니다",
            ["W3101"] = "한 프레임에서 찾은 영역 수가 상한에 걸려 잘렸습니다",
            ["W4101"] = "메모리 부족으로 완화 단계를 적용했습니다",
            ["W4102"] = "확신이 기준에 못 미쳐 원본 화소를 그대로 두었습니다",
            ["W4103"] = "안전 규칙에 따라 시간 창을 줄였습니다",
            ["W5101"] = "컨테이너 호환을 위해 스트림 하나를 빼고 저장했습니다",
            ["W5102"] = "스트림 복사를 쓸 수 없어 다시 인코딩했습니다",
            ["W6101"] = "다른 백엔드로 대체했습니다",
        }.ToFrozenDictionary(StringComparer.Ordinal);

    /// <summary>
    /// The Korean line for a code, or the protocol's own English meaning when there is none.
    /// </summary>
    /// <remarks>
    /// Falling back rather than throwing: a worker newer than this host may report a code this
    /// table has never seen, and losing the whole failure over its wording would be worse than
    /// showing it in English.
    /// </remarks>
    public static string Describe(ErrorCode code)
    {
        ArgumentNullException.ThrowIfNull(code);

        return Korean.TryGetValue(code.Code, out var text) ? text : code.Meaning;
    }

    /// <summary>The code and its meaning, as one line for the queue's detail column.</summary>
    public static string Line(ErrorCode code)
    {
        ArgumentNullException.ThrowIfNull(code);

        // The number stays: it is what docs/TROUBLESHOOTING.md and a search are keyed on.
        return $"{code.Code} · {Describe(code)}";
    }

    /// <summary>Codes this table has a Korean line for. For the completeness test.</summary>
    public static IReadOnlyCollection<string> Translated => Korean.Keys;
}
