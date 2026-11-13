import torch
from sgscfnet.models import build_model
from sgscfnet.losses import compute_loss, expected_safety_cost, diversity_penalty, ordinal_loss


def test_all_model_forms_and_backward():
    torch.manual_seed(10)
    images = torch.rand(4, 3, 48, 48)
    freshness = torch.tensor([0,1,2,1])
    species = torch.tensor([0,1,0,1])
    fw, sw, jw = torch.ones(3), torch.ones(2), torch.ones(6)
    for kind in ["freshness","multitask","joint24","sgscfnet-hard","sgscfnet-soft"]:
        model = build_model(kind, "tiny_cnn", 2, pretrained=False)
        output = model(images)
        assert output["freshness_logits"].shape == (4,3)
        loss = compute_loss(output, freshness, species, kind, fw, sw, jw)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_asymmetric_penalty():
    y = torch.tensor([2])
    mislabeled_acceptable = torch.tensor([[8.,0.,-8.]])
    correctly_not_fresh = torch.tensor([[-8.,0.,8.]])
    assert expected_safety_cost(mislabeled_acceptable, y) > expected_safety_cost(correctly_not_fresh, y)
    assert ordinal_loss(torch.randn(3,2), torch.tensor([0,1,2])) >= 0
    assert diversity_penalty(torch.randn(4,3,3)).ndim == 0


def test_gating_routes():
    model = build_model("sgscfnet-hard", "tiny_cnn", 3, pretrained=False)
    with torch.no_grad():
        model.species_head.weight.zero_()
        model.species_head.bias.copy_(torch.tensor([-4.0,7.0,-4.0]))
    out = model(torch.rand(2,3,40,40))
    assert (out["routing"].argmax(1) == 1).all()
    assert out["expert_logits"].shape == (2,3,3)
