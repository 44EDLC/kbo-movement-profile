import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import gradio as gr

try:
    import koreanize_matplotlib
except Exception:
    koreanize_matplotlib = None


# =========================================================
# 기본 설정
# =========================================================

PLAYER_DATA_DIR = "player_data"
PLAYER_LOOKUP_FILE = "player_lookup.csv"
PARK_EFFECT_FILE = "park_effects_slot_v2_by_season.csv"

plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic"]
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 선수 Lookup
# =========================================================

player_lookup = pd.read_csv(
    PLAYER_LOOKUP_FILE
)

player_lookup["player_name"] = (
    player_lookup["player_name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

player_lookup = player_lookup[
    player_lookup["player_name"] != ""
].copy()

player_lookup["pitcher_trackman_id"] = (
    player_lookup["pitcher_trackman_id"]
    .astype(int)
)

PLAYER_IDS = (
    player_lookup
    .sort_values(
        "pitches",
        ascending=False
    )
    .drop_duplicates(
        subset="player_name"
    )
    .set_index(
        "player_name"
    )["pitcher_trackman_id"]
    .to_dict()
)

print(
    f"선수 Lookup 로드 완료: "
    f"{len(player_lookup):,}명"
)


# =========================================================
# 구장 보정치
# =========================================================

effects = pd.read_csv(
    PARK_EFFECT_FILE
)

effects = effects[
    [
        "season",
        "park",
        "IVB_effect",
        "HB_effect"
    ]
].copy()


# =========================================================
# 데이터 열
# =========================================================

cols = [
    "season",
    "top_bottom",
    "pitcher_trackman_id",
    "pitcher_team",
    "batter_team",
    "pitcher_hand",
    "batter_hand",
    "tagged_pitch_type",
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
]


# =========================================================
# 구장
# =========================================================

PARK_MAP = {
    "KIW_HER": "Gocheok",
    "DOO_BEA": "Jamsil",
    "LG_TWI": "Jamsil",
    "SAM_LIO": "Daegu",
    "SK_WYV": "Incheon",
    "SSG_LAN": "Incheon",
    "LOT_GIA": "Sajik",
    "HAN_EAG": "Daejeon",
    "NC_DIN": "Changwon",
    "KT_WIZ": "Suwon",
}


# =========================================================
# 팀 이름
# =========================================================

TEAM_NAME_MAP = {
    "KIW_HER": "키움 히어로즈",
    "DOO_BEA": "두산 베어스",
    "LG_TWI": "LG 트윈스",
    "SAM_LIO": "삼성 라이온즈",
    "SK_WYV": "SK 와이번스",
    "SSG_LAN": "SSG 랜더스",
    "LOT_GIA": "롯데 자이언츠",
    "HAN_EAG": "한화 이글스",
    "NC_DIN": "NC 다이노스",
    "KT_WIZ": "KT 위즈",
    "KIA_TIG": "KIA 타이거즈",
}


HAND_MAP = {
    "Right": "RHP",
    "Left": "LHP",
    "R": "RHP",
    "L": "LHP",
}


# =========================================================
# 구종명
# =========================================================

PITCH_NAME_MAP = {
    "Fastball": "4-Seam",

    "Sinker": "Sinker",
    "SInker": "Sinker",

    "Cutter": "Cutter",
    "Slider": "Slider",
    "Curveball": "Curveball",

    "ChangeUp": "Changeup",
    "Changeup": "Changeup",

    "Splitter": "Splitter",
    "Sweeper": "Sweeper",
    "Slurve": "Slurve",
}


# =========================================================
# 구종 색상
# =========================================================

PITCH_COLORS = {
    "4-Seam": "#ff2d8d",
    "Changeup": "#57cc83",
    "Slider": "#f2cc2f",
    "Curveball": "#4357ff",
    "Cutter": "#b05ca8",
    "Sinker": "#aa176c",
    "Splitter": "#16a394",
    "Sweeper": "#24b869",
    "Slurve": "#6e87d8",
}


# =========================================================
# 타자 손잡이
# =========================================================

def normalize_batter_hand(x):

    if pd.isna(x):
        return np.nan

    x = str(x).strip().upper()

    if x in ["L", "LEFT", "LH"]:
        return "LHH"

    if x in ["R", "RIGHT", "RH"]:
        return "RHH"

    return np.nan


# =========================================================
# ★ 선수 한 명 데이터만 불러오기
# =========================================================

def load_player_data(player_id):

    file_path = os.path.join(
        PLAYER_DATA_DIR,
        f"{player_id}.parquet"
    )

    if not os.path.exists(file_path):
        return None

    player_df = pd.read_parquet(
        file_path,
        columns=cols
    )

    # -----------------------------------------------------
    # 홈팀 / 구장
    # -----------------------------------------------------

    player_df["home_team"] = np.where(
        player_df["top_bottom"] == "Top",
        player_df["pitcher_team"],
        player_df["batter_team"]
    )

    player_df["park"] = (
        player_df["home_team"]
        .map(PARK_MAP)
    )


    # -----------------------------------------------------
    # Movement cm → inch
    # -----------------------------------------------------

    player_df["IVB_raw"] = (
        pd.to_numeric(
            player_df["induced_vert_break"],
            errors="coerce"
        )
        / 2.54
    )

    player_df["HB_raw"] = (
        pd.to_numeric(
            player_df["horz_break"],
            errors="coerce"
        )
        / 2.54
    )


    # -----------------------------------------------------
    # 시즌별 구장 보정
    # -----------------------------------------------------

    player_df = player_df.merge(
        effects,
        on=["season", "park"],
        how="left"
    )

    player_df["IVB"] = (
        player_df["IVB_raw"]
        - player_df["IVB_effect"]
    )

    player_df["HB"] = (
        player_df["HB_raw"]
        - player_df["HB_effect"]
    )


    # -----------------------------------------------------
    # 타자 손잡이
    # -----------------------------------------------------

    player_df["batter_side"] = (
        player_df["batter_hand"]
        .apply(normalize_batter_hand)
    )

    return player_df


# =========================================================
# Movement Report
# =========================================================

def movement_report(
    player_name,
    season
):

    if player_name not in PLAYER_IDS:

        print(
            f"{player_name}의 TrackMan ID가 없습니다."
        )

        return None


    player_id = (
        PLAYER_IDS[player_name]
    )


    # =====================================================
    # 이 선수 데이터만 RAM으로 불러오기
    # =====================================================

    player_df = load_player_data(
        player_id
    )


    if player_df is None:

        print(
            "선수 데이터 파일이 없습니다."
        )

        return None


    p = player_df[
        player_df["season"]
        == season
    ].copy()


    if len(p) == 0:

        print(
            "해당 시즌 데이터가 없습니다."
        )

        return None


    # =====================================================
    # 구종 정리
    # =====================================================

    p["pitch_name"] = (
        p["tagged_pitch_type"]
        .map(PITCH_NAME_MAP)
    )


    p = p[
        p["pitch_name"].notna()
    ].copy()


    p = p[
        p["park"].notna()
        & p["IVB"].notna()
        & p["HB"].notna()
        & p["rel_speed"].notna()
    ].copy()


    # =====================================================
    # 최소 10구 구종
    # =====================================================

    counts = (
        p.groupby("pitch_name")
        .size()
    )


    valid_pitch_types = (
        counts[
            counts >= 10
        ].index
    )


    p = p[
        p["pitch_name"]
        .isin(valid_pitch_types)
    ].copy()


    total_pitches = len(p)


    if total_pitches == 0:

        print(
            "10구 이상인 구종 데이터가 없습니다."
        )

        return None


    # =====================================================
    # Movement 이상치
    # =====================================================

    clean = p[
        (p["IVB"].abs() <= 40)
        &
        (p["HB"].abs() <= 40)
    ].copy()


    # =====================================================
    # Arsenal
    # =====================================================

    arsenal = (
        clean
        .groupby("pitch_name")
        .agg(
            Count=(
                "pitch_name",
                "size"
            ),

            Velocity=(
                "rel_speed",
                "mean"
            ),

            IVB=(
                "IVB",
                "mean"
            ),

            HB=(
                "HB",
                "mean"
            ),

            Spin=(
                "spin_rate",
                "mean"
            )
        )
    )


    arsenal["Pitch%"] = (
        arsenal["Count"]
        / total_pitches
        * 100
    )


    arsenal = arsenal[
        [
            "Count",
            "Pitch%",
            "Velocity",
            "IVB",
            "HB",
            "Spin"
        ]
    ]


    arsenal = (
        arsenal.sort_values(
            "Pitch%",
            ascending=False
        )
    )


    pitch_order = (
        arsenal.index.tolist()
    )


    # =====================================================
    # 팀
    # =====================================================

    team_mode = (
        p["pitcher_team"]
        .dropna()
        .mode()
    )


    if len(team_mode) > 0:

        raw_team = (
            team_mode.iloc[0]
        )

    else:

        raw_team = ""


    team_name = (
        TEAM_NAME_MAP.get(
            raw_team,
            raw_team
        )
    )


    # =====================================================
    # 투수 손
    # =====================================================

    hand_mode = (
        p["pitcher_hand"]
        .dropna()
        .mode()
    )


    if len(hand_mode) > 0:

        raw_hand = (
            hand_mode.iloc[0]
        )

    else:

        raw_hand = ""


    hand_name = (
        HAND_MAP.get(
            raw_hand,
            raw_hand
        )
    )


    # =====================================================
    # LHH / RHH Usage
    # =====================================================

    usage = (
        p.groupby(
            [
                "pitch_name",
                "batter_side"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )


    if "LHH" not in usage.columns:
        usage["LHH"] = 0


    if "RHH" not in usage.columns:
        usage["RHH"] = 0


    total_lhh = (
        usage["LHH"].sum()
    )

    total_rhh = (
        usage["RHH"].sum()
    )


    if total_lhh > 0:

        usage["LHH_pct"] = (
            usage["LHH"]
            / total_lhh
            * 100
        )

    else:

        usage["LHH_pct"] = 0


    if total_rhh > 0:

        usage["RHH_pct"] = (
            usage["RHH"]
            / total_rhh
            * 100
        )

    else:

        usage["RHH_pct"] = 0


    usage = usage.reindex(
        pitch_order
    )


    # =====================================================
    # Summary
    # =====================================================

    avg_velo = (
        p["rel_speed"].mean()
    )

    avg_spin = (
        p["spin_rate"].mean()
    )


    lhh_count = int(
        (
            p["batter_side"]
            == "LHH"
        ).sum()
    )


    rhh_count = int(
        (
            p["batter_side"]
            == "RHH"
        ).sum()
    )


    # =====================================================
    # FIGURE
    # =====================================================

    fig = plt.figure(
        figsize=(19, 13.5),
        facecolor="white"
    )


    gs = GridSpec(
        19,
        12,
        figure=fig,

        left=0.045,
        right=0.97,

        top=0.975,
        bottom=0.045,

        hspace=1.05,
        wspace=0.62
    )


    # =====================================================
    # HEADER
    # =====================================================

    ax_header = fig.add_subplot(
        gs[0:3, :]
    )

    ax_header.axis("off")


    ax_header.text(
        0.5,
        0.86,
        player_name,

        ha="center",
        va="center",

        fontsize=36,
        fontweight="bold",

        color="#152033"
    )


    ax_header.text(
        0.5,
        0.60,
        f"{team_name}   |   {hand_name}",

        ha="center",
        va="center",

        fontsize=17,
        fontweight="medium",

        color="#556174"
    )


    ax_header.text(
        0.5,
        0.35,
        "Season Movement Summary",

        ha="center",
        va="center",

        fontsize=21,
        fontweight="bold",

        color="#152033"
    )


    ax_header.text(
        0.5,
        0.13,
        f"{season} KBO Season",

        ha="center",
        va="center",

        fontsize=14,
        fontweight="medium",

        color="#8a919c"
    )


    # =====================================================
    # SUMMARY TABLE
    # =====================================================

    ax_summary = fig.add_subplot(
        gs[3:5, :]
    )

    ax_summary.axis("off")


    summary_headers = [
        "Pitches",
        "Pitch Types",
        "Avg Velo",
        "Avg Spin",
        "vs LHH",
        "vs RHH"
    ]


    summary_values = [
        f"{total_pitches:,}",
        str(len(arsenal)),
        f"{avg_velo:.1f}",
        f"{avg_spin:.0f}",
        f"{lhh_count:,}",
        f"{rhh_count:,}"
    ]


    summary_table = (
        ax_summary.table(
            cellText=[
                summary_headers,
                summary_values
            ],

            cellLoc="center",

            bbox=[
                0.025,
                0.04,
                0.95,
                0.92
            ]
        )
    )


    summary_table.auto_set_font_size(
        False
    )


    for col in range(
        len(summary_headers)
    ):

        head = (
            summary_table[
                (0, col)
            ]
        )

        value = (
            summary_table[
                (1, col)
            ]
        )


        head.set_facecolor(
            "#17396d"
        )

        head.get_text().set_color(
            "white"
        )

        head.get_text().set_fontweight(
            "bold"
        )

        head.get_text().set_fontsize(
            13.5
        )


        value.set_facecolor(
            "#ffffff"
        )

        value.get_text().set_color(
            "#111827"
        )

        value.get_text().set_fontweight(
            "bold"
        )

        value.get_text().set_fontsize(
            15
        )


        head.set_linewidth(1.1)
        value.set_linewidth(1.1)


    # =====================================================
    # 그래프
    # =====================================================

    ax_velo = fig.add_subplot(
        gs[6:13, 0:4]
    )

    ax_break = fig.add_subplot(
        gs[6:13, 4:8]
    )

    ax_usage = fig.add_subplot(
        gs[6:13, 8:12]
    )


    # =====================================================
    # VELOCITY
    # =====================================================

    ax_velo.set_title(
        "Pitch Velocity Distribution",
        fontsize=18,
        fontweight="bold",
        pad=14
    )


    velocity_y = (
        np.arange(
            len(pitch_order)
        )[::-1]
        * 1.4
    )


    velo_min = max(
        80,
        np.floor(
            p["rel_speed"].min()
            - 4
        )
    )


    velo_max = min(
        165,
        np.ceil(
            p["rel_speed"].max()
            + 4
        )
    )


    for y, pitch in zip(
        velocity_y,
        pitch_order
    ):

        vals = (
            p.loc[
                p["pitch_name"]
                == pitch,
                "rel_speed"
            ]
            .dropna()
        )


        color = (
            PITCH_COLORS.get(
                pitch,
                "#777777"
            )
        )


        bins = np.linspace(
            velo_min,
            velo_max,
            65
        )


        hist, edges = (
            np.histogram(
                vals,
                bins=bins,
                density=True
            )
        )


        if len(hist) >= 5:

            kernel = np.array(
                [
                    1,
                    2,
                    3,
                    2,
                    1
                ],
                dtype=float
            )

            kernel /= (
                kernel.sum()
            )

            hist = np.convolve(
                hist,
                kernel,
                mode="same"
            )


        centers = (
            edges[:-1]
            + edges[1:]
        ) / 2


        if hist.max() > 0:

            hist = (
                hist
                / hist.max()
                * 0.95
            )


        ax_velo.fill_between(
            centers,
            y,
            y + hist,

            color=color,
            alpha=0.32
        )


        ax_velo.plot(
            centers,
            y + hist,

            color=color,
            linewidth=2.2
        )


        ax_velo.vlines(
            vals.mean(),

            y,
            y + 0.95,

            color=color,

            linestyle="--",
            linewidth=1.3
        )


    ax_velo.set_xlim(
        velo_min,
        velo_max
    )


    ax_velo.set_ylim(
        -0.45,
        max(velocity_y) + 1.35
    )


    ax_velo.set_yticks([])


    ax_velo.set_xlabel(
        "Velocity (km/h)",
        fontsize=13,
        fontweight="medium"
    )


    ax_velo.tick_params(
        axis="x",
        labelsize=11
    )


    ax_velo.grid(
        axis="x",
        alpha=0.12
    )


    # =====================================================
    # MOVEMENT
    # =====================================================

    ax_break.set_title(
        "Pitch Breaks",
        fontsize=18,
        fontweight="bold",
        pad=14
    )


    for radius in [
        6,
        12,
        18,
        24
    ]:

        ax_break.add_patch(
            Circle(
                (0, 0),
                radius,

                fill=False,

                linestyle="--",
                linewidth=0.85,

                edgecolor="#9ba3ad",

                alpha=0.55
            )
        )


    ax_break.axhline(
        0,

        color="#d78968",

        linestyle="--",
        linewidth=1,

        alpha=0.7
    )


    ax_break.axvline(
        0,

        color="#d78968",

        linestyle="--",
        linewidth=1,

        alpha=0.7
    )


    for pitch in pitch_order:

        pitch_data = clean[
            clean["pitch_name"]
            == pitch
        ]


        color = (
            PITCH_COLORS.get(
                pitch,
                "#777777"
            )
        )


        ax_break.scatter(
            pitch_data["HB"],
            pitch_data["IVB"],

            s=11,

            color=color,
            alpha=1.0,

            edgecolors="none",

            zorder=3
        )


    ax_break.set_xlim(
        -30,
        30
    )

    ax_break.set_ylim(
        -30,
        30
    )


    ax_break.set_aspect(
        "equal",
        adjustable="box"
    )


    ax_break.set_xlabel(
        "Horizontal Break (in)",
        fontsize=13,
        fontweight="medium"
    )


    ax_break.set_ylabel(
        "Induced Vertical Break (in)",
        fontsize=13,
        fontweight="medium"
    )


    ax_break.tick_params(
        axis="both",
        labelsize=11
    )


    ax_break.grid(
        alpha=0.10
    )


    # =====================================================
    # USAGE
    # =====================================================

    ax_usage.set_title(
        "Pitch Usage",
        fontsize=18,
        fontweight="bold",
        pad=14
    )


    usage_y = (
        np.arange(
            len(pitch_order)
        )[::-1]
        * 1.4
    )


    lhh = (
        usage.loc[
            pitch_order,
            "LHH_pct"
        ]
        .values
    )


    rhh = (
        usage.loc[
            pitch_order,
            "RHH_pct"
        ]
        .values
    )


    usage_colors = [
        PITCH_COLORS.get(
            pitch,
            "#777777"
        )

        for pitch in pitch_order
    ]


    ax_usage.barh(
        usage_y,
        -lhh,

        height=0.82,

        color=usage_colors
    )


    ax_usage.barh(
        usage_y,
        rhh,

        height=0.82,

        color=usage_colors
    )


    ax_usage.axvline(
        0,

        color="#27364b",
        linewidth=1.25
    )


    for yy, left, right in zip(
        usage_y,
        lhh,
        rhh
    ):

        if left >= 0.5:

            ax_usage.text(
                -left - 1.5,
                yy,

                f"{left:.1f}%",

                ha="right",
                va="center",

                fontsize=11,
                fontweight="medium",

                color="#4b5563"
            )


        if right >= 0.5:

            ax_usage.text(
                right + 1.5,
                yy,

                f"{right:.1f}%",

                ha="left",
                va="center",

                fontsize=11,
                fontweight="medium",

                color="#4b5563"
            )


    max_usage = max(
        55,

        np.ceil(
            max(
                lhh.max(),
                rhh.max()
            ) / 10
        ) * 10 + 5
    )


    ax_usage.set_xlim(
        -max_usage,
        max_usage
    )


    ax_usage.set_ylim(
        -0.6,
        max(usage_y) + 1.25
    )


    ax_usage.set_yticks([])


    ax_usage.set_xlabel(
        "Usage (%)",
        fontsize=13,
        fontweight="medium"
    )


    ax_usage.tick_params(
        axis="x",
        labelsize=11
    )


    ax_usage.text(
        -max_usage / 2,
        max(usage_y) + 0.75,

        "vs LHH",

        ha="center",

        fontsize=12,
        fontweight="bold"
    )


    ax_usage.text(
        max_usage / 2,
        max(usage_y) + 0.75,

        "vs RHH",

        ha="center",

        fontsize=12,
        fontweight="bold"
    )


    ax_usage.grid(
        axis="x",
        alpha=0.12
    )


    # =====================================================
    # ARSENAL TABLE
    # =====================================================

    ax_table = fig.add_subplot(
        gs[14:19, :]
    )

    ax_table.axis("off")


    columns = [
        "Pitch Name",
        "Count",
        "Pitch%",
        "Velocity",
        "IVB",
        "HB",
        "Spin"
    ]


    table_data = []
    row_colors = []


    for pitch, row in arsenal.iterrows():

        table_data.append([
            pitch,

            f"{int(row['Count']):,}",

            f"{row['Pitch%']:.1f}%",

            f"{row['Velocity']:.1f}",

            f"{row['IVB']:.1f}",

            f"{row['HB']:.1f}",

            f"{row['Spin']:.0f}"
        ])


        row_colors.append(
            PITCH_COLORS.get(
                pitch,
                "#cccccc"
            )
        )


    table = ax_table.table(
        cellText=table_data,

        colLabels=columns,

        cellLoc="center",

        bbox=[
            0.025,
            0.13,
            0.95,
            0.84
        ]
    )


    table.auto_set_font_size(
        False
    )


    # =====================================================
    # TABLE HEADER - ARIAL
    # =====================================================

    for col in range(
        len(columns)
    ):

        cell = (
            table[
                (0, col)
            ]
        )


        cell.set_facecolor(
            "#17396d"
        )


        cell.get_text().set_color(
            "white"
        )


        cell.get_text().set_fontfamily(
            "Arial"
        )


        cell.get_text().set_fontweight(
            "bold"
        )


        cell.get_text().set_fontsize(
            13
        )


        cell.set_linewidth(
            1.15
        )


    # =====================================================
    # TABLE DATA - ARIAL
    # =====================================================

    for row_i in range(
        1,
        len(table_data) + 1
    ):

        for col_i in range(
            len(columns)
        ):

            cell = (
                table[
                    (row_i, col_i)
                ]
            )


            cell.get_text().set_fontfamily(
                "Arial"
            )


            cell.get_text().set_fontsize(
                12.5
            )


            cell.get_text().set_fontweight(
                "semibold"
            )


            cell.set_linewidth(
                1.0
            )


            if col_i == 0:

                color = (
                    row_colors[
                        row_i - 1
                    ]
                )


                cell.set_facecolor(
                    color
                )


                cell.get_text().set_fontweight(
                    "bold"
                )


                if color in [
                    "#f2cc2f",
                    "#57cc83"
                ]:

                    cell.get_text().set_color(
                        "#111111"
                    )


                else:

                    cell.get_text().set_color(
                        "white"
                    )


            else:

                cell.set_facecolor(
                    "#f5f6f8"
                )


                cell.get_text().set_color(
                    "#161b22"
                )


    ax_table.text(
        0.025,
        0.035,

        "Arm-slot matched park correction applied",

        transform=ax_table.transAxes,

        fontsize=10.5,
        fontweight="medium",

        color="#667085"
    )


    # =====================================================
    # 저장
    # =====================================================

    os.makedirs(
        "reports",
        exist_ok=True
    )


    filename = (
        f"reports/"
        f"{player_name}_{season}_movement_report.png"
    )


    plt.savefig(
        filename,

        dpi=220,

        facecolor="white"
    )


    # 웹 서버에서는 창을 띄우지 않음
    plt.close(fig)


    # 선수 데이터도 더 이상 필요 없음
    del player_df
    del p
    del clean


    print(
        f"{player_name} {season} "
        f"프로필 생성 완료"
    )


    return filename


# =========================================================
# 웹사이트용 선수 목록
# =========================================================

PLAYER_NAMES = sorted(
    player_lookup["player_name"]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[
        lambda x: x != ""
    ]
    .unique()
    .tolist()
)


# =========================================================
# 선수 선택 → 시즌
# =========================================================

def update_seasons(
    player_name
):

    if not player_name:

        return gr.Dropdown(
            choices=[],
            value=None
        )


    matches = player_lookup[
        player_lookup["player_name"]
        == player_name
    ].copy()


    if len(matches) == 0:

        return gr.Dropdown(
            choices=[],
            value=None
        )


    selected = (
        matches
        .sort_values(
            "pitches",
            ascending=False
        )
        .iloc[0]
    )


    player_id = int(
        selected[
            "pitcher_trackman_id"
        ]
    )


    file_path = os.path.join(
        PLAYER_DATA_DIR,
        f"{player_id}.parquet"
    )


    if not os.path.exists(
        file_path
    ):

        return gr.Dropdown(
            choices=[],
            value=None
        )


    # 시즌 열 하나만 읽음
    season_df = pd.read_parquet(
        file_path,
        columns=["season"]
    )


    seasons = sorted(
        season_df["season"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


    del season_df


    if len(seasons) == 0:

        return gr.Dropdown(
            choices=[],
            value=None
        )


    return gr.Dropdown(
        choices=seasons,
        value=seasons[-1]
    )


# =========================================================
# 웹 프로필 생성
# =========================================================

def generate_profile_web(
    player_name,
    season
):

    if not player_name:

        raise gr.Error(
            "투수를 선택해주세요."
        )


    if season is None:

        raise gr.Error(
            "시즌을 선택해주세요."
        )


    matches = player_lookup[
        player_lookup["player_name"]
        == player_name
    ].copy()


    if len(matches) == 0:

        raise gr.Error(
            "선수를 찾을 수 없습니다."
        )


    selected = (
        matches
        .sort_values(
            "pitches",
            ascending=False
        )
        .iloc[0]
    )


    player_id = int(
        selected[
            "pitcher_trackman_id"
        ]
    )


    PLAYER_IDS[player_name] = (
        player_id
    )


    season = int(
        season
    )


    filename = movement_report(
        player_name,
        season
    )


    if filename is None:

        raise gr.Error(
            "프로필 생성에 실패했습니다."
        )


    return filename


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="KBO Movement Profile Generator"
) as demo:

    gr.Markdown(
        """
# ⚾ KBO Movement Profile Generator

KBO TrackMan 데이터를 이용한 투수 Movement Profile입니다.

**투수를 선택하고 시즌을 고른 뒤 프로필 생성 버튼을 눌러주세요.**
        """
    )


    with gr.Row():

        player_input = gr.Dropdown(
            choices=PLAYER_NAMES,
            label="투수",
            value=None,
            filterable=True,
            interactive=True
        )


        season_input = gr.Dropdown(
            choices=[],
            label="시즌",
            value=None,
            interactive=True
        )


    generate_button = gr.Button(
        "프로필 생성",
        variant="primary"
    )


    output_image = gr.Image(
        label="Movement Profile",
        type="filepath"
    )


    player_input.change(
        fn=update_seasons,
        inputs=player_input,
        outputs=season_input
    )


    generate_button.click(
        fn=generate_profile_web,
        inputs=[
            player_input,
            season_input
        ],
        outputs=output_image
    )


# =========================================================
# 서버
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            7860
        )
    )


    demo.launch(
        server_name="0.0.0.0",
        server_port=port
    )