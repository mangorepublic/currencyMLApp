package com.example.myapplication;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;

//import com.example.myapplication.ml.Model1;

public class MainActivity extends AppCompatActivity {

    public static final int CAMERA_ACTION_CODE=1;
    public static final int REQUEST_IMAGE_CAPTURE=1;
    private static final int MAX_RESULTS = 4 ;
    ImageView Imageview;
    Button capture;
    Bitmap bitmap;
    ByteBuffer byteBuffer;
    TensorImage tbuffer;
    TextView textview;
    private int imageSizeX;
    private int imageSizeY;
    private TensorBuffer outputProbabilityBuffer;
    TensorProcessor probabilityProcessor;
    MappedByteBuffer tfliteModel;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;
    List<String> Nlabels;

    TextToSpeech t1;

    ArrayList<Float> Entries = new ArrayList<>();

    Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Imageview = findViewById(R.id.View_id);
        capture = findViewById(R.id.imageBtn);
        textview = findViewById(R.id.textView);


        t1 = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status != TextToSpeech.ERROR){
                  int lang =  t1.setLanguage(Locale.ENGLISH);
                }
            }
        });

        try{
            tflite=new Interpreter(loadModelFile());
        }catch (Exception e) {
            e.printStackTrace();
        }


        capture.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                askCameraPermission();
            }
        });
    }

    private void askCameraPermission() {
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, 101);
        }
        else{
            openCamera();
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        //if(intent.resolveActivity(getPackageManager()) != null)
        try{
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
        }
        catch(ActivityNotFoundException e){
            Toast.makeText(MainActivity.this, "There is no app that support this action",Toast.LENGTH_SHORT).show();
        }


    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 101 && grantResults.length < 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
            openCamera();

        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            Bitmap imageBitmap = (Bitmap) extras.get("data");
            Imageview.setImageBitmap(imageBitmap);
            bitmap = Bitmap.createScaledBitmap(imageBitmap,180,180,true);


            int imageTensorIndex = 0;
            int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
            imageSizeY = imageShape[1];
            imageSizeX = imageShape[2];
            DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

            int probabilityTensorIndex = 0;
            int[] probabilityShape =
                    tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
            DataType probabilityDataType = tflite.getOutputTensor(probabilityTensorIndex).dataType();

            tbuffer = new TensorImage(imageDataType);
            outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
            probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();

            tbuffer = loadImage(bitmap);

            tflite.run(tbuffer.getBuffer(),outputProbabilityBuffer.getBuffer().rewind());
            
            showResults();

    }


   // @Override
    //protected void onActivityResult(int requestCode, int resultCode, Intent data) {
      //  super.onActivityResult(requestCode, resultCode, data);
        //if(resultCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK && data != null){
            //Bundle bundle = data.getExtras();
          //  Bitmap finalPhoto = (Bitmap) data.getExtras().get("data");
            //Imageview.setImageBitmap(finalPhoto);

        //    }

      //  }
    //}
        }



    private TensorImage loadImage(final Bitmap imagebitmap) {
        // Loads bitmap into a TensorImage.
        tbuffer.load(imagebitmap);
        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(getPreprocessNormalizeOp())
                        .build();
        return imageProcessor.process(tbuffer);

            }
    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private TensorOperator getPostprocessNormalizeOp(){
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }






    private List<String> LoadLabelList() throws IOException{
        List<String> labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(this.getAssets().open("label.txt")));
        String line;
        while((line = reader.readLine()) != null){
            labels.add(line);
        }
        reader.close();
        return labels;
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("tf_lite_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

    }


    

    private void showResults() {
        try{
            Nlabels = FileUtil.loadLabels(MainActivity.this,"label.txt");
        }catch (Exception e){
            e.printStackTrace();
        }
        Map<String, Float> labeledProbability =
                new TensorLabel(Nlabels, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();
        float maxValueInMap =(Collections.max(labeledProbability.values()));

        for (Map.Entry<String, Float> entry : labeledProbability.entrySet()) {
            if (entry.getValue()==maxValueInMap) {
                String[] label = labeledProbability.keySet().toArray(new String[0]);
                Float[] label_probability = labeledProbability.values().toArray(new Float[0]);

                String new_str = label[0];
                Float max = label_probability[0];

                for(int i=0; i<label_probability.length; i++)
                {
                    if(label_probability[i] > max){
                        max = label_probability[i];
                        new_str = label[i];
                    }
                }

                int speech = t1.speak(new_str,TextToSpeech.QUEUE_FLUSH,null);


                textview.setText("Predictions:" + label[0] + " and " + String.valueOf(label_probability[0])
                        + label[1] + " and " + String.valueOf(label_probability[1])
                        + label[2] + " and " + String.valueOf(label_probability[2])
                        + label[3] + " and " + String.valueOf(label_probability[3])
                        + "   and " + String.valueOf(max)
                        + "     and " + new_str);
            }}}}

