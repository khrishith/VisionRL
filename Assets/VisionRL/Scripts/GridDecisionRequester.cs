using Unity.MLAgents;

namespace VisionRL
{
    // ML-Agents 4.0.0's requester subscribes in Awake and does not check enabled.
    // Respect the component toggle so offline manual control has no idle actions.
    public sealed class GridDecisionRequester : DecisionRequester
    {
        protected override bool ShouldRequestDecision(DecisionRequestContext context)
            => isActiveAndEnabled && base.ShouldRequestDecision(context);
        protected override bool ShouldRequestAction(DecisionRequestContext context)
            => isActiveAndEnabled && base.ShouldRequestAction(context);
    }
}
