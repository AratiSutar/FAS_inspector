import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app.ocr.engine import OCRResult
from app.detector.shape_detector import DetectionResult
from app.verifier.cross_verifier import verify, VerificationStatus


def make_ocr(variant=None, token=None, conf=0.9, raw=None):
    return OCRResult(
        raw_texts=raw or [],
        matched_variant=variant,
        matched_token=token,
        confidence=conf,
    )


def make_det(cls=None, conf=0.85, bbox=None):
    return DetectionResult(
        predicted_class=cls,
        confidence=conf,
        bbox=bbox,
        all_predictions=[],
    )


def test_match_su61855():
    result = verify(make_ocr("SU61855", "DSU61855"), make_det("SU61855"))
    assert result.status == VerificationStatus.MATCH
    assert result.passed is True


def test_match_su66072():
    result = verify(make_ocr("SU66072RAW", "SUG6072"), make_det("SU66072RAW"))
    assert result.status == VerificationStatus.MATCH


def test_mismatch():
    result = verify(make_ocr("SU61855", "DSU61855"), make_det("SU66072RAW"))
    assert result.status == VerificationStatus.MISMATCH
    assert result.passed is False


def test_mismatch_different_bell_housings():
    result = verify(make_ocr("SU61856", "SUG1856"), make_det("SU61855"))
    assert result.status == VerificationStatus.MISMATCH


def test_no_code():
    result = verify(make_ocr(None, None, 0.0, ["SRF", "INDIA"]), make_det("SU61855"))
    assert result.status == VerificationStatus.NO_CODE
    assert result.detected_shape == "SU61855"


def test_no_shape():
    result = verify(make_ocr("SU61855", "DSU61855"), make_det(None, 0.0))
    assert result.status == VerificationStatus.NO_SHAPE


def test_low_ocr_conf():
    result = verify(make_ocr("SU61855", "DSU61855", conf=0.3), make_det("SU61855", conf=0.8))
    assert result.status == VerificationStatus.LOW_CONFIDENCE


def test_low_detector_conf():
    result = verify(make_ocr("SU61855", "DSU61855", conf=0.9), make_det("SU61855", conf=0.2))
    assert result.status == VerificationStatus.LOW_CONFIDENCE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])