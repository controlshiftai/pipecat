from pipecat.utils.enums import EndTaskReason


def test_transfer_terminal_reasons_are_stable_values():
    assert EndTaskReason.TRANSFERRED.value == "transferred"
    assert EndTaskReason.TRANSFER_FAILED.value == "transfer_failed"
