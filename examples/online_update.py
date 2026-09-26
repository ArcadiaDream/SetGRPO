"""Apply a certified provider to one repeated GRPO rollout batch."""

from setgrpo.online import build_update


def main() -> None:
    def provider(group_id, scores, metadata):
        # Replace this callback with FrozenCertificateProvider in production.
        return [-0.3, -0.1, 0.1, 0.3]

    update = build_update(
        [0.1, 0.2, 0.3, 0.4],
        ["group-0"] * 4,
        rollout_size=4,
        provider=provider,
    )
    print(update.coefficients, update.acceptance_rate)


if __name__ == "__main__":
    main()
