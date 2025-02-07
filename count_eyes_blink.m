clear all;

% 目の検出オブジェクト
eyeDetector = vision.CascadeObjectDetector('EyePairBig');

% ポイントトラッカーオブジェクト
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% 動画を読み込む
videoName = "example-videoName"; % 読み込む動画の名前
v = VideoReader(videoName + ".mp4");

% 動画プレイヤーの作成
videoPlayer = vision.VideoPlayer('Position', [100 100 [v.Width / 2, v.Height / 2] + 30]);
videoPlayerZoom = vision.DeployableVideoPlayer();

runLoop = true;
numPts = 0;
frameCount = 0;
blinkCount = 0;  % 瞬きのカウント
eyeClosed = false;

% 輝度閾値の設定（初期値）
prevEyeIntensity = 255;  % 目の領域の前の輝度
blinkThreshold = 1;  % 瞬き検出のための輝度変化のしきい値

while runLoop && hasFrame(v)
    % 次のフレームを取得
    videoFrame = readFrame(v);
    videoFrameGray = rgb2gray(videoFrame);  % グレースケールに変換
    frameCount = frameCount + 1;

    if numPts < 10
        % 目の検出モード
        bboxeye = eyeDetector.step(videoFrameGray);

        if ~isempty(bboxeye)
            % 検出された目の領域内で特徴点を検出
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', bboxeye(1, :));

            % ポイントトラッカーの初期化
            xyPoints = points.Location;
            numPts = size(xyPoints,1);
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);

            % 特徴点の保存
            oldPoints = xyPoints;

            % 目の領域を四角形として変換
            bboxPoints = bbox2points(bboxeye(1, :));
            bboxPolygon = reshape(bboxPoints', 1, []);

            % 目の領域をフレームに描画
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, xyPoints, '+', 'Color', 'white');
        end
    else
        % 特徴点追跡モード
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);

        numPts = size(visiblePoints, 1);

        if numPts >= 10
            % 特徴点間の幾何変換を推定
            [xform, inlierIdx] = estimateGeometricTransform2D(oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
            oldInliers = oldInliers(inlierIdx, :);
            visiblePoints = visiblePoints(inlierIdx, :);

            % 目の領域を変換
            bboxPoints = transformPointsForward(xform, bboxPoints);
            bboxPolygon = reshape(bboxPoints', 1, []);

            % 目の領域をフレームに描画
            videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, 'LineWidth', 3);
            videoFrame = insertMarker(videoFrame, visiblePoints, '+', 'Color', 'white');

            % 目の領域を抽出
            eyeRegion = videoFrameGray(bboxeye(1,2):bboxeye(1,2)+bboxeye(1,4), bboxeye(1,1):bboxeye(1,1)+bboxeye(1,3));
            
            % 目の輝度の平均値を計算
            eyeIntensity = mean(eyeRegion(:));
             disp(['abs(prevEyeIntensity - eyeIntensity): ', num2str(abs(prevEyeIntensity - eyeIntensity))]);
            % 輝度変化に基づいて瞬きを検出
            if abs(prevEyeIntensity - eyeIntensity) > blinkThreshold  % 輝度の変化が閾値を超えるとき
                if eyeIntensity < prevEyeIntensity  % 輝度が低くなった場合、目が閉じた
                    if ~eyeClosed
                        blinkCount = blinkCount + 1;
                        disp(['Blink Count: ', num2str(blinkCount)]);
                        eyeClosed = true;
                    end
                else
                    eyeClosed = false;
                end
            end

            % 目の輝度を更新
            prevEyeIntensity = eyeIntensity;

            % ポイントを更新
            oldPoints = visiblePoints;
            setPoints(pointTracker, oldPoints);
        end
    end
    videoFrame = insertText(videoFrame, [0 50],  ['Eye Closed ' num2str(blinkCount,'%d') ' times'], 'FontSize', 30);
    
    % 動画フレームを表示
    step(videoPlayer, videoFrame);
    
    pause(0.1);
    % ウィンドウが閉じられたかチェック
    runLoop = isOpen(videoPlayer);
end

disp(['Blink Count: ', num2str(blinkCount)]);

% クリーンアップ
clear v;
release(videoPlayer);
release(pointTracker);
release(eyeDetector);
