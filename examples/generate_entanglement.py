"""Generate a tiny suite3 entanglement example."""

from hawkesnest.suites import EntanglementSuite


def main() -> None:
    result = EntanglementSuite().generate(
        level="L2",
        n_events=50,
        seed=123,
        out_dir="outputs/examples/entanglement",
    )
    print(result.events.head())
    print(result.metadata)


if __name__ == "__main__":
    main()
