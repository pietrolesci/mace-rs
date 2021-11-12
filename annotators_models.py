import pyro
import pyro.distributions as dist
import torch
from torch import nn
import torch.nn.functional as F
from pyro.poutine import reparam
from pyro.infer.reparam import LocScaleReparam
from pyro.ops.indexing import Vindex


def multinomial(annotations: torch.Tensor) -> None:
    """
    This model corresponds to the plate diagram in Figure 1 of reference [1].

    Interpretation:
        For each class k, every annotator has the same ability \zeta_k of getting it right.
        In particular, if the correct class is k = c, we want each annotator to have a very
        skewed zeta_c vector putting the majority of the probability mass on c.

        The probability of the correct class c is parametrized by \pi. In this way we can
        have a measure of how difficult to label is each item. In this model every item has
        the same difficulty \pi. In general, easy items have a very skewed \pi.
    """
    num_classes = annotations.unique().numel()
    num_items, num_positions = annotations.shape

    with pyro.plate("class", num_classes):
        zeta = pyro.sample("zeta", dist.Dirichlet(torch.ones(num_classes)))

    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(num_classes)))

    with pyro.plate("item", num_items, dim=-2):
        c = pyro.sample("c", dist.Categorical(pi))

        with pyro.plate("position", num_positions):
            pyro.sample("y", dist.Categorical(zeta[c]), obs=annotations)


def dawid_skene(positions: torch.Tensor, annotations: torch.Tensor) -> None:
    """
    This model corresponds to the plate diagram in Figure 2 of reference [1].
    """
    num_annotators = positions.unique().numel()
    num_classes = annotations.unique().numel()
    num_items, num_positions = annotations.shape

    with pyro.plate("annotator", num_annotators, dim=-2):
        with pyro.plate("class", num_classes):
            beta = pyro.sample("beta", dist.Dirichlet(torch.ones(num_classes)))

    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(num_classes)))

    with pyro.plate("item", num_items, dim=-2):
        c = pyro.sample("c", dist.Categorical(pi))

        # here we use Vindex to allow broadcasting for the second index `c`
        # ref: http://num.pyro.ai/en/latest/utilities.html#pyro.contrib.indexing.vindex
        with pyro.plate("position", num_positions):
            pyro.sample("y", dist.Categorical(Vindex(beta)[positions, c, :]), obs=annotations)


def mace(positions: torch.Tensor, annotations: torch.Tensor) -> None:
    """
    This model corresponds to the plate diagram in Figure 3 of reference [1].
    """
    num_annotators = positions.unique().numel()
    num_classes = annotations.unique().numel()
    num_items, num_positions = annotations.shape

    with pyro.plate("annotator", num_annotators):
        epsilon = pyro.sample("epsilon", dist.Dirichlet(torch.full((num_classes,), 10, dtype=torch.float32)))
        theta = pyro.sample("theta", dist.Beta(0.5, 0.5))

    with pyro.plate("item", num_items, dim=-2):
        # NB: using constant logits for discrete uniform prior
        # (pyro does not have DiscreteUniform distribution yet)
        c = pyro.sample("c", dist.Categorical(logits=torch.zeros(num_classes)))

        with pyro.plate("position", num_positions):
            s = pyro.sample("s", dist.Bernoulli(1 - theta[positions]))
            probs = torch.where(s[..., None] == 0, F.one_hot(c, num_classes).type(epsilon.dtype), epsilon[positions])
            pyro.sample("y", dist.Categorical(probs), obs=annotations)


def hierarchical_dawid_skene(positions: torch.Tensor, annotations: torch.Tensor) -> None:
    """
    This model corresponds to the plate diagram in Figure 4 of reference [1].
    """
    num_annotators = positions.unique().numel()
    num_classes = annotations.unique().numel()
    num_items, num_positions = annotations.shape

    with pyro.plate("class", num_classes):
        # NB: we define `beta` as the `logits` of `y` likelihood; but `logits` is
        # invariant up to a constant, so we'll follow [1]: fix the last term of `beta`
        # to 0 and only define hyperpriors for the first `num_classes - 1` terms.
        zeta = pyro.sample("zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        omega = pyro.sample("Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))

    with pyro.plate("annotator", num_annotators, dim=-2):
        with pyro.plate("class_abilities", num_classes):
            # non-centered parameterization
            with reparam(config={"beta": LocScaleReparam(centered=None)}):
                beta = pyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
            # pad 0 to the last item
            beta = F.pad(beta, [(0, 0)] * (beta.dim() - 1) + [(0, 1)])

    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(num_classes)))

    with pyro.plate("item", num_items, dim=-2):
        c = pyro.sample("c", dist.Categorical(pi))

        with pyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :]
            pyro.sample("y", dist.Categorical(logits=logits), obs=annotations)


def item_difficulty(annotations: torch.Tensor) -> None:
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_classes = annotations.unique().numel()
    num_items, _ = annotations.shape

    with pyro.plate("class", num_classes):
        eta = pyro.sample("eta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        chi = pyro.sample("Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))

    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(num_classes)))

    with pyro.plate("item", num_items, dim=-2):
        c = pyro.sample("c", dist.Categorical(pi))

        with reparam(config={"theta": LocScaleReparam(centered=None, shape_params=0)}):
            theta = pyro.sample("theta", dist.Normal(eta[c], chi[c]).to_event(1))
            theta = F.pad(theta, [(0, 0)] * (torch.ndim(theta) - 1) + [(0, 1)])

        with pyro.plate("position", annotations.shape[-1]):
            pyro.sample("y", dist.Categorical(logits=theta), obs=annotations)


def logistic_random_effects(positions: torch.Tensor, annotations: torch.Tensor) -> None:
    """
    This model corresponds to the plate diagram in Figure 5 of reference [1].
    """
    num_annotators = positions.unique().numel()
    num_classes = annotations.unique().numel()
    num_items, num_positions = annotations.shape

    with pyro.plate("class", num_classes):
        zeta = pyro.sample("zeta", dist.Normal(0, 1).expand([num_classes - 1]).to_event(1))
        omega = pyro.sample("Omega", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))
        chi = pyro.sample("Chi", dist.HalfNormal(1).expand([num_classes - 1]).to_event(1))

    with pyro.plate("annotator", num_annotators, dim=-2):
        with pyro.plate("class", num_classes):
            with reparam(config={"beta": LocScaleReparam(0)}):
                beta = pyro.sample("beta", dist.Normal(zeta, omega).to_event(1))
                beta = F.pad(beta, [(0, 0)] * (torch.ndim(beta) - 1) + [(0, 1)])

    pi = pyro.sample("pi", dist.Dirichlet(torch.ones(num_classes)))

    with pyro.plate("item", num_items, dim=-2):
        c = pyro.sample("c", dist.Categorical(pi))

        with reparam(config={"theta": LocScaleReparam(0)}):
            theta = pyro.sample("theta", dist.Normal(0, chi[c]).to_event(1))
            theta = F.pad(theta, [(0, 0)] * (torch.ndim(theta) - 1) + [(0, 1)])

        with pyro.plate("position", num_positions):
            logits = Vindex(beta)[positions, c, :] - theta
            pyro.sample("y", dist.Categorical(logits=logits), obs=annotations)
