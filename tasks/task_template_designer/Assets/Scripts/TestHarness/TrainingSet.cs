using System.Diagnostics;
public class TrainingSet : GameLevelSetInfo{

    public bool trainingTerminated; 

    public TrainingSet():base(){
        base.hasTimeLimit = false;
        base.hasInteractionLimit = false;
        base.hasLevelAttepmtLimit = false;
        base.hasLevelTimeLimit = false;
        base.notifyNovelty = false;
        base.currentLevelIndex = 0;
        base.mustSolve = false;
        trainingTerminated = false;
    }
}