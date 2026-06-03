"""Generate a tiny suite4 heterogeneity example."""

from hawkesnest.suites import HeterogeneitySuite


def main() -> None:
    result = HeterogeneitySuite().generate(
        level="H3",
        n_events=50,
        seed=123,
        out_dir="outputs/examples/heterogeneity",
    )
    print(result.events.head())
    print(result.metadata)


if __name__ == "__main__":
    main()
