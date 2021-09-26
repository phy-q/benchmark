public class TestSet : GameLevelSetInfo{
    public TestSet():base(){
        
        base.hasTimeLimit = false;
        base.hasInteractionLimit = false;
        base.hasLevelAttepmtLimit = false;
        base.hasLevelTimeLimit = false;
        base.notifyNovelty = false;
        base.currentLevelIndex = 0;
        base.mustSolve = false;
    }
}